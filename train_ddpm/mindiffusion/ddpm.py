from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from abc import ABC, abstractmethod
import numpy as np
from util.utils import accumulate
from gms.diffusion.lib.model import UNet
from gms.diffusion.lib.diffusion import GaussianDiffusion, make_beta_schedule
from gms.diffusion.lib.samplers import get_time_sampler
from torch import nn, optim
from resizer import Resizer


class DDPM(nn.Module):
    def __init__(
            self,
            eps_model: nn.Module,
            betas: Tuple[float, float],
            n_T: int,
            criterion: nn.Module = nn.MSELoss(), relative_complexity=None,
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        self.relative_complexity = relative_complexity
        self.beta_0 = betas[0]
        self.beta_1 = betas[1]

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def generate_ts(self, label):
        result_list = []
        numpy_label = label.cpu().detach().numpy()
        for i in numpy_label.tolist():
            if i == self.relative_complexity[-1]:
                range_tau = self.relative_complexity.index(i)
                tau_item = random.uniform(range_tau * 0.1, range_tau * 0.1 + 0.1)
                result_list.append(tau_item)
            else:
                range_tau = 0
                tau_item = random.uniform(range_tau * 0.1, range_tau * 0.1 + 0.1)
                result_list.append(tau_item)
        # tensor_tau_0 = torch.Tensor(result_list * 1000).cuda()     # [int(i*1000) for i in result_list]
        tensor_tau_0 = torch.Tensor([int(i * 1000) for i in result_list]).cuda().long()
        return tensor_tau_0

    def forward(self, x: torch.Tensor, label):  # -> torch.Tensor
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """
        # 改变时间的分布
        _ts = self.generate_ts(label)

        # _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        score = self.eps_model(x_t, _ts / self.n_T)
        mix_paras = self.sqrtmab[_ts, None, None, None] * x_t
        a = 0.7
        paras = a * mix_paras + (1 - a) * score
        p_loss = torch.square(paras - eps)
        return self.criterion(eps, paras), p_loss, _ts / self.n_T

    def sample(self, n_sample: int, size, device, end=None, x_1=None) -> torch.Tensor:

        # x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        x_i = x_1
        if end is None:
            end = 0

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, end, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            )
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
        x_i = torch.clamp(x_i, min=-3., max=3.)
        return x_i


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """

    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


def my_ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    from functools import partial
    to_torch = partial(torch.tensor, dtype=torch.float32)
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    v_posterior = 0.8
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    alphas_cumprod = torch.cumprod(alpha_t, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
    posterior_variance = (1 - v_posterior) * beta_t * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + v_posterior * beta_t
    posterior_mean_coef1 = beta_t * torch.sqrt(alphas_cumprod_prev) / (1. - to_torch(alphas_cumprod))
    posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alpha_t) / (1. - to_torch(alphas_cumprod))
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DiffusionBase(ABC):
    """
    Abstract base class for all diffusion implementations.
    """

    def __init__(self, args):
        super().__init__()
        self.sigma2_0 = args.sigma2_0
        self.sde_type = args.sde_type

    @abstractmethod
    def f(self, t):
        """ returns the drift coefficient at time t: f(t) """
        pass

    @abstractmethod
    def g2(self, t):
        """ returns the squared diffusion coefficient at time t: g^2(t) """
        pass

    @abstractmethod
    def var(self, t):
        """ returns variance at time t, \sigma_t^2"""
        pass

    @abstractmethod
    def e2int_f(self, t):
        """ returns e^{\int_0^t f(s) ds} which corresponds to the coefficient of mean at time t. """
        pass

    @abstractmethod
    def inv_var(self, var):
        """ inverse of the variance function at input variance var. """
        pass

    @abstractmethod
    def mixing_component(self, x_noisy, var_t, t, enabled):
        """ returns mixing component which is the optimal denoising model assuming that q(z_0) is N(0, 1) """
        pass

    @abstractmethod
    def integral_beta(self, t):
        pass

    @abstractmethod
    def antiderivative(self, t, stabilizing_constant=0.):
        pass

    @abstractmethod
    def normalizing_constant(self, t_min):
        pass

    @abstractmethod
    def get_diffusion_time(self, batch_size, batch_device, t_min):
        pass


class DiffusionGeometric(DiffusionBase):
    """
    Diffusion implementation with dz = -0.5 * beta(t) * z * dt + sqrt(beta(t)) * dW SDE and geometric progression of
    variance. This is our new diffusion.
    """

    def __init__(self, args):
        super().__init__(args)
        self.sigma2_min = args.sigma2_min
        self.sigma2_max = args.sigma2_max

    def f(self, t):
        return -0.5 * self.g2(t)

    def g2(self, t):
        sigma2_geom = self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t)
        log_term = np.log(self.sigma2_max / self.sigma2_min)
        return sigma2_geom * log_term / (1.0 - self.sigma2_0 + self.sigma2_min - sigma2_geom)

    def var(self, t):
        return self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t) - self.sigma2_min + self.sigma2_0

    def e2int_f(self, t):
        return torch.sqrt(
            1.0 + self.sigma2_min * (1.0 - (self.sigma2_max / self.sigma2_min) ** t) / (1.0 - self.sigma2_0))

    def inv_var(self, var):
        return torch.log((var + self.sigma2_min - self.sigma2_0) / self.sigma2_min) / np.log(
            self.sigma2_max / self.sigma2_min)

    def mixing_component(self, x_noisy, var_t, t, enabled):
        if enabled:
            return torch.sqrt(var_t) * x_noisy
        else:
            return None


class DDP(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.C = C
        # self.save_hyperparameters(conf)
        self.shape = (16, 16, 16)

        predict_var = True
        self.model = UNet(in_channel=16, channel=128, channel_multiplier=(1, 2, 2, 2), n_res_blocks=2,
                          attn_strides=[16, ],
                          dropout=0.1,
                          fold=1, predict_var=predict_var)
        self.ema = UNet(in_channel=16, channel=128, channel_multiplier=(1, 2, 2, 2), n_res_blocks=2,
                        attn_strides=[16, ],
                        dropout=0.1,
                        fold=1, predict_var=predict_var)
        self.betas = make_beta_schedule(type=C.beta_schedule, start=1e-4, end=2e-2, n_timestep=1000)
        self.diffusion = GaussianDiffusion(betas=self.betas, model_mean_type="eps", model_var_type="fixedsmall",
                                           loss_type="mse")
        self.sampler = get_time_sampler(sampler_type="loss-second-moment")(self.diffusion)
        self.optimizer_type = 'adam'
        self.lr = 2e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.relative_complexity = None
        self.p_sample_loop_progressive = C.p_sample_loop_progressive
        self.paper = C.paper
        self.shape = (C.bs, 3, C.image_size, C.image_size)
        self.shape_d = (C.bs, 3, int(C.image_size / C.down_N), int(C.image_size / C.down_N))
        self.down = Resizer(self.shape, 1 / C.down_N).to("cuda")
        self.up = Resizer(self.shape_d, C.down_N).to("cuda")
        self.resizers = (self.down, self.up)
        self.range_t = C.range_t

    def forward(self, size, end=None, x_1=None):
        if self.p_sample_loop_progressive:
            return \
                self.diffusion.p_sample_loop_progressive(self.model, size, resizers=self.resizers,
                                                         range_t=self.range_t, paper=self.paper)[0]
        else:
            return self.diffusion.p_sample_loop(self.model, size, end=end, x_1=x_1)

    def training_step(self, batch, batch_ix):
        img = batch
        # time = torch.randint(size=(img.shape[0],), low=0, high=self.conf.model.schedule.n_timestep,
        #                      dtype=torch.int64, device=img.device)
        if self.paper:
            time = self.sampler.generate_ts(batch_ix, self.relative_complexity)
            _, weights = self.sampler.sample(img.size(0), device=img.device)
        else:
            time, weights = self.sampler.sample(img.size(0), device=img.device)
        loss, l2_term = self.diffusion.training_losses(self.model, img, time)
        self.sampler.update_with_all_losses(time, loss)
        p_loss = l2_term * weights.reshape(-1, 1, 1, 1)
        loss = torch.mean(loss * weights)
        accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model, 0.9999)

        # self.log('train_loss', loss, logger=True)
        return {'loss': loss, "p_loss": p_loss, "ts": time, "weights": weights}

    def validation_step(self, batch, batch_ix):
        img, _ = batch
        time, weights = self.sampler.sample(img.size(0), device=img.device)
        loss = self.diffusion.training_losses(self.model, img, time)
        loss = torch.mean(loss * weights)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        # self.log('val_loss', avg_loss, logger=True, prog_bar=True, on_epoch=True)
        return {'val_loss': avg_loss}
