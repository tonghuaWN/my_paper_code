import torch
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from gms import utils
from train_ddpm.mindiffusion.unet import NaiveUnet
from train_ddpm.mindiffusion.ddpm import DDPM
from train_ddpm.mindiffusion.unet import Conv3
import numpy as np


@torch.jit.script
def sample_normal_jit(mu, sigma):
    rho = mu.mul(0).normal_()
    z = rho.mul_(sigma).add_(mu)
    return z, rho


class Normal:
    def __init__(self, mu, log_sigma):
        self.mu = mu
        self.log_sigma = log_sigma
        self.sigma = torch.exp(log_sigma)

    def sample(self, t=1.):
        return sample_normal_jit(self.mu, self.sigma * t)

    def sample_given_rho(self, rho):
        return rho * self.sigma + self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - self.log_sigma
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(self.log_sigma) + normal_dist.log_sigma

    def mean(self):
        return self.mu


class VAE(utils.GM):
    DC = utils.AttrDict()  # default C
    DC.z_size = 128
    DC.beta = 1.0

    def __init__(self, C):
        super().__init__(C)
        self.encoder = Encoder(C.z_size, C)
        self.decoder = Decoder(C.z_size, C)
        self.optimizer = Adam(self.parameters(), lr=C.lr)
        self.NaiveUnet = NaiveUnet(8, 8, n_feat=128)
        self.ddpm = DDPM(eps_model=self.NaiveUnet, betas=(1e-4, 0.02), n_T=1000).to("cuda:0")
        self.ddpm_optim = torch.optim.Adam(self.ddpm.parameters(), lr=1e-5)

    def input_for_diffusion(self, x):
        features = self.encoder.net(x)
        mu, log_std = self.encoder.get_mu_log_std(features)
        z = self.encoder.reparameterize(mu, log_std)
        z = z.reshape(z.shape[0], z.shape[1], 1, 1)
        input_diffusion = self.encoder.vae_diffusion(z)
        z1 = self.decoder.decode_net(input_diffusion)
        return input_diffusion, mu, log_std, z1

    def compute_kl(self, x, y):
        import torch.nn as nn

        x = F.log_softmax(x)
        y = F.softmax(y, dim=1)
        criterion = nn.KLDivLoss()
        klloss = criterion(x, y)
        return klloss

    def sum_log_q(self, all_log_q):
        log_q = 0.
        for log_q_conv in all_log_q:
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])

        return log_q

    def get_log_q(self, mu, log_std):
        dist = Normal(mu, log_std)
        eps, _ = dist.sample()
        log_q_conv = dist.log_p(eps)
        return log_q_conv

    def loss(self, x):
        input_diffusion, mu, log_std, z = self.input_for_diffusion(x)
        ddpm_loss, score = self.ddpm(input_diffusion)
        decoded = self.decoder(z)
        recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean((1, 2, 3))
        # z_prior = tdib.Normal(0, 1)
        z_prior = torch.rand_like(input_diffusion)
        a = 0.8
        z_ddpm_prior = a * z_prior + (1 - a) * score
        kl_loss = self.compute_kl(z_ddpm_prior, input_diffusion) * self.C.beta  # z_ddpm_prior  input_diffusion
        q_loss = torch.mean(recon_loss) + kl_loss
        metrics = {'vae_loss': q_loss, 'recon_loss': recon_loss.mean(), 'kl_loss': kl_loss.mean(),
                   "ddpm_loss": ddpm_loss}
        return q_loss, metrics, ddpm_loss

        pass

    # def loss(self, x):
    #     """VAE loss"""
    #     z_post = self.encoder(x)  # posterior  p(z|x)
    #     decoded = self.decoder(z_post.rsample())  # reconstruction p(x|z)
    #     recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean((1, 2, 3))
    #     # kl div constraint
    #     z_prior = tdib.Normal(0, 1)
    #     # z_ddpm_prior=a*z_prior+(1-a)*ddpm_loss
    #     kl_loss = tdib.kl_divergence(z_post, z_prior).mean(-1)
    #     # full loss and metrics
    #     loss = (recon_loss + self.C.beta * kl_loss).mean()
    #     metrics = {'vae_loss': loss, 'recon_loss': recon_loss.mean(), 'kl_loss': kl_loss.mean()}
    #     return loss, metrics

    def sample(self, n):
        z = th.randn(n, self.C.z_size).to(self.C.device)
        # z = th.randn(n,16,8,8).to(self.C.device)
        return self._decode(z)

    # def evaluate(self, writer, x, epoch):
    #     """run samples and other evaluations"""
    #     samples = self.sample(25)
    #     writer.add_image('samples', utils.combine_imgs(samples, 5, 5)[None], epoch)
    #     z_post = self.encoder(x[:8])
    #     recon = self._decode(z_post.mean)
    #     recon = th.cat([x[:8].cpu(), recon], 0)
    #     writer.add_image('reconstruction', utils.combine_imgs(recon, 2, 8)[None], epoch)

    def evaluate(self, writer, x, epoch):
        """run samples and other evaluations"""
        z_recon = self.ddpm.sample(25, (8, 32, 32), "cuda:0")
        decoder_input = self.decoder.decode_net(z_recon)
        samples = self._decode(decoder_input)
        writer.add_image('samples', utils.combine_imgs(samples, 5, 5)[None], epoch)
        input_diffusion, mu, log_std, z1 = self.input_for_diffusion(x[:8])
        recon = self.decoder.decode_net(input_diffusion)
        # z_post = self.encoder(x[:8])
        recon = self._decode(recon)
        recon = th.cat([x[:8].cpu(), recon], 0)
        writer.add_image('reconstruction', utils.combine_imgs(recon, 2, 8)[None], epoch)

    def _decode(self, x):
        return 1.0 * (self.decoder(x).exp() > 0.5).cpu()


class Encoder(nn.Module):
    def __init__(self, out_size, C):
        super().__init__()
        self.drop_out = 0.1
        H = C.hidden_size
        self.net = nn.Sequential(
            nn.Conv2d(1, H, 3, 2),
            nn.ReLU(),
            nn.Conv2d(H, H, 3, 2),
            nn.ReLU(),
            nn.Conv2d(H, H, 3, 1),
            nn.ReLU(),
            nn.Conv2d(H, 2 * out_size, 3, 2),
            nn.Dropout(self.drop_out),
            nn.Flatten(1, 3),
        )
        self.diffusion_net = nn.Sequential(  # F S P
            nn.Conv2d(1, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, 2, 1),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )

        self.vae_diffusion = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            Conv3(64, 64),
            Conv3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2),
            Conv3(32, 32),
            Conv3(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, 2),
            Conv3(16, 16),
            Conv3(16, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, 2),
            Conv3(16, 16),
            Conv3(16, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, 2),
            Conv3(8, 8),
            Conv3(8, 8),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
        )

    def get_dist(self, x):
        # mu, log_std = x.chunk(2, -1)
        mu, log_std = x.chunk(2, 1)
        std = F.softplus(log_std) + 1e-4
        return tdib.Normal(mu, std)

    def reparameterize(self, mu, log_std):
        std = torch.exp(0.5 * log_std)
        eps = torch.rand_like(std)
        return eps * std + mu

    def get_mu_log_std(self, x):
        mu, log_std = x.chunk(2, 1)
        return mu, log_std

    def forward(self, x):
        # x = self.diffusion_net(x)
        # return self.get_dist(x)
        # x = self.net(x)
        # x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        # x = self.vae_diffusion(x)
        return self.get_dist(self.net(x))

    # def input_for_diffusion(self, x):
    #     features = self.net(x)
    #     mu, log_std = self.get_mu_log_std(features)
    #     z = self.reparameterize(mu, log_std)
    #     input_diffusion = self.vae_diffusion(z)
    #     return input_diffusion, mu, log_std, z

    def get_loss(self, x):
        input_diffusion, mu, log_std, z = self.input_for_diffusion(x)


class Decoder(nn.Module):
    def __init__(self, in_size, C):
        super().__init__()
        H = C.hidden_size
        self.drop_out = 0.1
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_size, H, 5, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(H, H, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(H, H, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(H, 1, 3, 1),
            nn.Dropout(self.drop_out),
        )
        # self.diffusion_net = nn.Sequential(
        #     nn.ConvTranspose2d(16, 16, 3, 2, 1),
        #     nn.ReLU(),  # b 16 14 14
        #     nn.ConvTranspose2d(16, 8, 3, 1, 1),
        #     nn.ReLU(),  # b 8 14 14  16
        #     nn.ConvTranspose2d(8, 4, 3, 1, 1),
        #     nn.ReLU(),  # b 4 14 14  16
        #     nn.ConvTranspose2d(4, 1, 4, 1, 2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(1, 1, 4, 2, 1),
        # )  # b 1 28 28

        self.decode_net = nn.Sequential(  # F S P
            nn.Conv2d(8, 32, 4, 2, 1),  # (B,32,16,16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Flatten(1, 3),
        )
        self.conv1 = residual_down_sampling_block(8, 32)
        self.conv2 = residual_down_sampling_block(32, 64)
        self.conv3 = residual_down_sampling_block(64, 128)
        self.conv4 = residual_down_sampling_block(128, 128)
        self.conv5 = residual_down_sampling_block(128, 128)

    def decode_help(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self.net(x[..., None, None])
        # x = self.diffusion_net(x)
        return x


class residual_down_sampling_block(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(residual_down_sampling_block, self).__init__()

        self.conv1_layer = nn.Conv2d(channel_in, channel_out // 2, kernel_size=3, stride=1, padding=1)
        self.BatchNorm_Layer1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2_layer = nn.Conv2d(channel_out // 2, channel_out, kernel_size=3, stride=1, padding=1)
        self.BatchNorm_Layer2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1)

        self.AvePool = nn.AvgPool2d(scale, scale)

    def forward(self, input):
        x = self.conv3(self.AvePool(input))

        result = F.rrelu(self.BatchNorm_Layer1(self.conv1_layer(input)))
        result = self.AvePool(result)
        result = self.BatchNorm_Layer2(self.conv2_layer(result))

        return F.rrelu(result + x)
