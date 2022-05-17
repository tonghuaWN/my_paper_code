import time
import pathlib
import argparse
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from itertools import count
import torch as th
from gms import utils
from gms import autoregs, vaes, gans, diffusion
import torch
from feature_extraction import relative_complexity

# TRAINING SCRIPT

C = utils.AttrDict()
C.model = 'vae'
C.bs = 400
C.hidden_size = 512
C.device = 'cuda'
C.num_epochs = 120
C.save_n = 100
C.logdir = pathlib.Path('./logs/')
C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
C.lr = 1e-4
C.class_cond = 0
C.binarize = 1
C.pad32 = 0
C.dataset = 'mnist'
C.image_size = 28 if C.dataset == 'mnist' else 32
C.test_freq = 5
C.num_x_bits = 8
C.relative_complexity = None
C.channel = 1 if C.dataset == 'mnist' else 3
C.reverse_test = True
C.beta_schedule = "cosine"
C.p_sample_loop_progressive = True
C.paper = True  # 测试论文相关代码
C.down_N = 32
C.range_t = 20

if __name__ == '__main__':
    # PARSE CMD LINE
    parser = argparse.ArgumentParser()
    for key, value in C.items():
        parser.add_argument(f'--{key}', type=utils.args_type(value), default=value)
    tempC, _ = parser.parse_known_args()
    # 设置随机种子
    torch.manual_seed(100)
    np.random.seed(100)
    torch.cuda.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    torch.backends.cudnn.benchmark = True
    # SETUP
    Model = {
        'rnn': autoregs.RNN,
        'made': autoregs.MADE,
        'wavenet': autoregs.Wavenet,
        'pixelcnn': autoregs.PixelCNN,
        'gatedcnn': autoregs.GatedPixelCNN,
        'transformer': autoregs.TransformerCNN,
        'vae': vaes.VAE,
        'vqvae': vaes.VQVAE,
        'gan': gans.GAN,
        'diffusion': diffusion.DiffusionModel,
    }[tempC.model]
    defaults = {'logdir': C.logdir / C.model}
    for key, value in Model.DC.items():
        defaults[key] = value
        if key not in tempC:
            parser.add_argument(f'--{key}', type=type(value), default=value)
    parser.set_defaults(**defaults)
    C = parser.parse_args()
    model = Model(C=C).to(C.device)
    writer = SummaryWriter(C.logdir)
    logger = utils.dump_logger({}, writer, 0, C)
    if C.dataset == 'mnist':
        train_ds, test_ds = utils.load_mnist(C.bs, binarize=C.binarize, pad32=C.pad32)
    elif C.dataset == 'cifar10':
        train_ds, test_ds = utils.load_cifar10(C.bs, binarize=C.binarize, pad32=C.pad32)
    else:
        train_ds = None
        test_ds = None
    num_vars = utils.count_vars(model)  # 计算模型参数量
    print('num_vars', num_vars)
    # 计算相对复杂度
    tau_list = relative_complexity(train_ds, C.num_x_bits, C.channel)
    print("复杂度的相对顺序关系，从左到右依次简单，高-->低:")
    print(tau_list)
    model.ddpm.relative_complexity = tau_list
    # TRAINING LOOP
    for epoch in count():
        # TRAIN
        train_time = time.time()
        for batch in train_ds:
            batch[0], batch[1] = batch[0].to(C.device), batch[1].to(C.device)
            # if C.dataset == 'cifar10':
            #     batch[0] = utils.symmetrize_image_data(batch[0])
            # TODO: see if we can just use loss and write the gan such that it works.
            metrics = model.train_step(batch[0], batch[1])
            for key in metrics:
                if key != "ts" and key != "weights":
                    logger[key] += [metrics[key].detach().cpu()]
            writer.add_histogram('ts', metrics["ts"], epoch)
            writer.add_histogram('weights', metrics["weights"], epoch)
        logger['dt/train'] = time.time() - train_time
        logger = utils.dump_logger(logger, writer, epoch, C)
        if (epoch + 1) % C.test_freq == 0:
            # TEST
            model.eval()
            with th.no_grad():
                # if we define an explicit loss function, use it to test how we do on the test set.
                if hasattr(model, 'loss'):
                    for test_batch in test_ds:
                        # if C.dataset == 'cifar10':
                        #     test_batch[0] = utils.symmetrize_image_data(test_batch[0])
                        test_batch[0], test_batch[1] = test_batch[0].to(C.device), test_batch[1].to(C.device)
                        test_loss, test_metrics, ddpm_loss = model.loss(test_batch[0], test_batch[1])
                        for key in test_metrics:
                            if key != "ts":
                                logger['test/' + key] += [test_metrics[key].detach().cpu()]
                else:
                    test_batch = next(iter(test_ds))
                    test_batch[0], test_batch[1] = test_batch[0].to(C.device), test_batch[1].to(C.device)
                # run the model specific evaluate function. usually draws samples and creates other relevant visualizations.
                eval_time = time.time()
                model.evaluate(writer, test_batch[0], epoch)
                logger['dt/evaluate'] = time.time() - eval_time
            model.train()
            # LOGGING
            logger['num_vars'] = num_vars
            logger = utils.dump_logger(logger, writer, epoch, C)
            if epoch % C.save_n == 0:
                path = C.logdir / 'model.pt'
                print("SAVED MODEL", path)
                th.save(model.state_dict(), path)
            if epoch >= C.num_epochs:
                break
