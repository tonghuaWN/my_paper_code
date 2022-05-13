from __future__ import print_function

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from nn import normal_init, NonLinear


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# =======================================================================================================================
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

    # AUXILIARY METHODS  备注：上述self.means、self.idle_input属性原本在add_pseudoinputs()函数中定义的，但没有运行该函数，就没有定义其函数中的属性
    def add_pseudoinputs(self):
        # create pseudo-input
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)  # 产生0-1之间的数字：<0,0;>1,1;x,其他
        self.means = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False,
                               activation=nonlinearity)
        # 参数：500,784

        # init pseudo-inputs
        if self.args.use_training_data_init:
            self.means.linear.weight.data = self.args.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)  #

        # create an idle input for calling pseudo-inputs　生成一个number_components＊number_components的对角单位矩阵
        self.idle_input = Variable(torch.eye(self.args.number_components, self.args.number_components),
                                   requires_grad=False)
        if self.args.cuda:
            self.idle_input = self.idle_input.cuda()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def calculate_loss(self):
        return 0.

    def calculate_likelihood(self):
        return 0.

    def calculate_lower_bound(self):
        return 0.

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        return 0.

# =======================================================================================================================
