from __future__ import print_function
import sys

sys.path.append('/home/ubuntu/桌面/vae_vpflows-master')
import numpy as np

import math

from scipy.misc import logsumexp

import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable

from distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from visual import plot_histogram
from nn import he_init, GatedDense, NonLinear

from Model import Model


def xavier_init(m):
    s = np.sqrt(2. / (m.in_features + m.out_features))
    m.weight.data.normal_(0, s)


class HF(nn.Module):
    def __init__(self):
        super(HF, self).__init__()

    def forward(self, v, z):
        '''
        :param v: batch_size (B) x latent_size (L)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = z - 2* v v_T / norm(v,2) * z
        '''
        # v * v_T：向量v与v的转置相乘 torch.bmm的使用：两个张量必须时三维的
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L
        # v * v_T * z
        vvTz = torch.bmm(vvT, z.unsqueeze(2)).squeeze(
            2)  # A * z : batchdot( B x L x L * B x L x 1 ).squeeze(2) = (B x L x 1).squeeze(2) = B x L
        # calculate norm ||v||^2
        norm_sq = torch.sum(v * v, 1).unsqueeze(1)  # calculate norm-2 for each row : B x 1

        norm_sq = norm_sq.expand(norm_sq.size(0), v.size(1))  # expand sizes : B x L
        # calculate new z
        z_new = z - 2 * vvTz / norm_sq  # z - 2 * v * v_T  * z / norm2(v)
        return z_new


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, h, g):
        return h * g


# =======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        # encoder: q(z | x)
        self.q_z_layers = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), 300),
            GatedDense(300, 300)
        )

        self.q_z_mean = Linear(300, self.args.z1_size)
        self.q_z_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers = nn.Sequential(
            GatedDense(self.args.z1_size, 300),
            GatedDense(300, 300)
        )

        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(300, np.prod(self.args.input_size), activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(300, np.prod(self.args.input_size), activation=nn.Sigmoid())
            self.p_x_logvar = NonLinear(300, np.prod(self.args.input_size),
                                        activation=nn.Hardtanh(min_val=-4.5, max_val=0))

        # Householder flow
        self.v_layers = nn.ModuleList()
        # T > 0
        if self.args.number_of_flows > 0:
            # T = 1
            self.v_layers.append(nn.Linear(300, self.args.z1_size))
            # T > 1
            for i in range(1, self.args.number_of_flows):
                self.v_layers.append(nn.Linear(self.args.z1_size, self.args.z1_size))

        self.sigmoid = nn.Sigmoid()
        self.Gate = Gate()
        self.HF = HF()

        # Xavier initialization (normal)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)
        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()

    # AUXILIARY METHODS
    def log_gmm(self, x, mean, log_var, average=False, dim=None):
        k = len(mean)
        log_gmm = 0.0
        for i in range(k):
            log_gmm += (1 / k) * torch.rsqrt(2 * math.pi * torch.exp(log_var[i])) * torch.exp(
                -0.5 * torch.pow(x[i] - mean[i], 2) / torch.exp(log_var[i]))

        log_gmm = torch.log(log_gmm)

        if average:
            return torch.mean(log_gmm, dim)
        else:
            return torch.sum(log_gmm, dim)

    def log_gmm_Normal_standard(self, x, average=False, dim=None):

        log_normal = -0.5 * math.log(2 * math.pi) - 0.5 * torch.pow(x, 2)
        if average:
            return torch.mean(log_normal, dim)
        else:
            return torch.sum(log_normal, dim)

    def calculate_loss(self, x, beta=1., average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        # pass through VAE
        x_mean, x_logvar, z_0, z_q_iid, z_T, z_q_mean, z_q_logvar = self.forward(x)

        # RE

        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)

        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)

        else:
            raise Exception('Wrong input type!')

        # KL
        '''
        log_p_z = self.log_gmm_Normal_standard(z_q,dim=1)
        log_q_z = self.log_gmm(z_q_iid, z_q_mean, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)
        '''
        '''
        KL = 0.0
        k = len(z_q_mean)
        for i in range(k):
            log_p_z = log_Normal_standard(z_q_iid[i], dim=1)
            log_q_z = log_Normal_diag(z_q_iid[i], z_q_mean[i], z_q_logvar[i], dim=1)
            KL += - (1/k)*(log_p_z - log_q_z)
        '''

        KL = 0
        k = len(z_q_mean)
        for i in range(k):
            KL += (1 / k) * torch.sum(z_q_logvar[i] - z_q_mean[i].pow(2) - z_q_logvar[i].exp(), 1)
            # KL += (1/k)*(torch.log(torch.prod(torch.exp(z_q_logvar[i])))-torch.sum(torch.exp(z_q_logvar[i]))-\
            #    torch.sum(z_q_mean[i]*z_q_mean[i],1))
        KL = -0.5 * self.args.z1_size - 0.5 * KL

        loss = - RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB=100):
        # set auxiliary variables for number of training and test sets
        N_test = X.size(0)

        # init list
        likelihood_test = []

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB

        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            # Take x*
            x_single = X[j].unsqueeze(0)  # size[1,784]

            a = []
            for r in range(0, int(R)):
                # Repeat it for all training points
                x = x_single.expand(S, x_single.size(1))

                a_tmp, _, _ = self.calculate_loss(x)

                a.append(-a_tmp.cpu().data.numpy())

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp(a)
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

        plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_test)

    def calculate_lower_bound(self, X_full, MB=100):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.

        I = int(math.ceil(X_full.size(0) / MB))
        for i in range(I):
            x = X_full[i * MB: (i + 1) * MB].view(-1, int(np.prod(self.args.input_size)))

            loss, RE, KL = self.calculate_loss(x, average=True)

            RE_all += RE.cpu().data[0]
            KL_all += KL.cpu().data[0]

            # CALCULATE LOWER-BOUND: RE + KL - ln(N)
            lower_bound += loss.cpu().data[0]

        lower_bound = lower_bound / I

        return lower_bound

    # ADDITIONAL METHODS
    def generate_x(self, N=25):

        z_sample_rand = Variable(torch.FloatTensor(N, self.args.z1_size).normal_())
        if self.args.cuda:
            z_sample_rand = z_sample_rand.cuda()

        samples_rand, _ = self.p_x(z_sample_rand)
        return samples_rand

    def reconstruct_x(self, x):
        x_mean, _, _, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x, k=1):
        '''
        self.q_z_mean = Linear(300, self.args.z1_size)
        self.q_z_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))
        '''
        x = self.q_z_layers(x)
        z_q_mean = []
        z_q_logvar = []
        for i in range(k):
            z_q_mean.append(self.q_z_mean(x))
            z_q_logvar.append(self.q_z_logvar(x))
        return z_q_mean, z_q_logvar, x

    # THE MODEL: HOUSEHOLDER FLOW
    def q_z_Flow(self, z, h_last):
        v = {}
        # Householder Flow:
        if self.args.number_of_flows > 0:
            v['1'] = self.v_layers[0](h_last)
            z['1'] = self.HF(v['1'], z['0'])
            for i in range(1, self.args.number_of_flows):
                v[str(i + 1)] = self.v_layers[i](v[str(i)])
                z[str(i + 1)] = self.HF(v[str(i + 1)], z[str(i)])
        return z

    def reparameterize(self, mu, logvar):
        '''
        :repara_sum:各分量加权之和
        :repara:各个独立分量
        '''
        k = len(mu)
        std = []
        eps = []
        repara = []
        repara_sum = 0
        for i in range(k):
            std_tmp = logvar[i].mul(0.5).exp_()
            std.append(std_tmp)

            if self.args.cuda:
                eps_tmp = torch.cuda.FloatTensor(std_tmp.size()).normal_()
            else:
                eps_tmp = torch.FloatTensor(std_tmp.size()).normal_()
            eps_tmp = Variable(eps_tmp)
            eps.append(eps_tmp)
        for i in range(k):
            repara_tmp = eps[i].mul(std[i]).add_(mu[i])
            repara.append(repara_tmp)
        for i in range(k):
            repara_sum += repara[i] * (1 / k)
        return repara_sum, repara

        # THE MODEL: GENERATIVE DISTRIBUTION

    def p_x(self, z):
        z = self.p_x_layers(z)

        x_mean = self.p_x_mean(z)
        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0. + 1. / 512., max=1. - 1. / 512.)
            x_logvar = self.p_x_logvar(z)
        return x_mean, x_logvar

    # the prior
    def log_p_z(self, z):

        return log_Normal_standard(z, dim=1)

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        z = {}
        # z ~ q(z | x)
        z_q_mean, z_q_logvar, h_last = self.q_z(x, k=self.args.number_of_gmm)
        z['0'], z_q_iid = self.reparameterize(z_q_mean, z_q_logvar)
        # Householder Flow:
        z = self.q_z_Flow(z, h_last)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z[str(self.args.number_of_flows)])

        return x_mean, x_logvar, z['0'], z_q_iid, z[str(self.args.number_of_flows)], z_q_mean, z_q_logvar
