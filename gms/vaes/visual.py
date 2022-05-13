import os

import numpy as np

import torch
from torch.autograd import Variable

import matplotlib

matplotlib.use('Agg')  # 新加入的两行代码，不加的话，会出先如下错误：Exception ignored in: <bound method Image.__del__ of
# <tkinter.PhotoImage object at 0x7f5d6c491128>>   RuntimeError: main thread is not in main loop
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =======================================================================================================================
def plot_histogram(x, dir, mode):
    fig = plt.figure()

    # the histogram of the data
    n, bins, patches = plt.hist(x, 100, density=True, facecolor='blue', alpha=0.5)  # normed=True  更改为density=True

    # plt.xlabel('Log-likelihood value')
    # plt.ylabel('Probability')
    plt.xlim(0, 250)
    plt.ylim(0, 0.03)
    plt.grid(True)

    plt.savefig(dir + 'histogram_' + mode + '.png', bbox_inches='tight')
    plt.close(fig)


'''
ax = plt.figure().add_subplot(111)

ax.plot(epoch, val_loss_list[0], color = 'blue', label = 'Standard VAE')
ax.plot(epoch, val_loss_list[1], color = 'brown', label = 'GpVAE+Pixel+Mix1')
ax.plot(epoch, val_loss_list[2], color = 'green', label = 'GpVAE+Pixel+Mix2')
ax.plot(epoch, val_loss_list[3], color = 'red', label = 'GpVAE+Pixel+Mix3')
ax.set_xlabel("Number of epochs")
ax.set_ylabel("loss function value")
plt.xlim(200,2000)
plt.ylim(65,100)
plt.legend(loc = 'best')
plt.grid()
#plt.show()
plt.savefig('a.jpg')
'''


# =======================================================================================================================
def plot_images(args, x_sample, dir, file_name, size_x=3, size_y=3):
    fig = plt.figure(figsize=(size_x, size_y))
    # fig = plt.figure(1)
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # plt.imshow(sample.reshape(args.input_size[1], args.input_size[2],-1), cmap='Greys_r')
        # plt.imshow(sample.reshape(args.input_size[1], args.input_size[2], -1))  # 绘制彩色图像
        plt.imshow(sample.reshape(args.input_size[0], args.input_size[1], args.input_size[2]).transpose([1, 2, 0]))  # 参数要通用化

    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)


# =======================================================================================================================
def plot_real(args, x_sample, dir, size_x=3, size_y=3):
    x_sample = x_sample.data.cpu().numpy()[:size_x * size_y]

    fig = plt.figure(figsize=(size_x, size_y))
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(args.input_size[1], args.input_size[2]), cmap='Greys_r')

    plt.savefig(dir + 'real.png', bbox_inches='tight')
    plt.close(fig)


# =======================================================================================================================
def plot_reconstruction(args, samples, c, dir, size_x=3, size_y=3):
    samples = samples.data.cpu().numpy()[:size_x * size_y]

    fig = plt.figure(figsize=(size_x, size_y))
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(args.input_size[1], args.input_size[2]), cmap='Greys_r')

    if not os.path.exists(dir + 'reconstruction/'):
        os.makedirs(dir + 'reconstruction/')

    plt.savefig(dir + 'reconstruction/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
    plt.close(fig)


# =======================================================================================================================
def plot_scatter(args, model, X, Y, dir, limit=None):
    z_mean, z_logvar = model.q_z(X)
    if args.model_name == 'vae' or 'scorevae':
        z_sample = model.reparameterize(z_mean, z_logvar)
    else:
        z_sample, _ = model.reparameterize(z_mean, z_logvar)
    Z = z_sample.data.cpu().numpy()

    Y = Y.data.cpu().numpy()
    if len(np.shape(Y)) > 2:
        Y_label = np.argmax(Y, 1)
    else:
        Y_label = Y

    fig = plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=Y_label, alpha=0.5, edgecolors='k', cmap='gist_ncar')
    plt.colorbar()

    if limit != None:
        # set axes range
        limit = np.abs(limit)
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)

    plt.savefig(dir + 'scatter2D.png', bbox_inches='tight')
    plt.close(fig)


# =======================================================================================================================
def plot_scatter_VPflow(model, args, X, Y, dir, limit=None):
    z = {}
    z_mean, z_logvar, h_last = model.q_z(X)
    z['0'] = model.reparameterize(z_mean, z_logvar)
    z = model.q_z_Flow(z, h_last)
    Z = z[str(args.number_of_flows)].data.cpu().numpy()

    Y = Y.data.cpu().numpy()
    if len(np.shape(Y)) > 2:
        Y_label = np.argmax(Y, 1)
    else:
        Y_label = Y

    fig = plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=Y_label, alpha=0.5, edgecolors='k', cmap='gist_ncar')
    plt.colorbar()

    if limit != None:
        # set axes range
        limit = np.abs(limit)
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)

    plt.savefig(dir + 'scatter2D.png', bbox_inches='tight')
    plt.close(fig)


# =======================================================================================================================
def plot_scatter2(model, Z, Y, dir, name='scatter2D.png', limit=None):
    Z = Z.data.cpu().numpy()

    Y = Y.data.cpu().numpy()
    if len(np.shape(Y)) > 2:
        Y_label = np.argmax(Y, 1)
    else:
        Y_label = Y

    fig = plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=Y_label, alpha=0.5, edgecolors='k', cmap='gist_ncar')
    plt.colorbar()

    if limit != None:
        # set axes range
        limit = np.abs(limit)
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)

    plt.savefig(dir + name, bbox_inches='tight')
    plt.close(fig)


# =======================================================================================================================
def plot_generation(args, samples_mean, dir, size_x=3, size_y=3):
    # decode
    samples = samples_mean.data.cpu().numpy()[:size_x * size_y]

    fig = plt.figure(figsize=(size_x, size_y))
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(args.input_size[1], args.input_size[2]), cmap='Greys_r')

    plt.savefig(dir + 'generation.png', bbox_inches='tight')
    plt.close(fig)


# =======================================================================================================================
def plot_manifold(model, args, dir, x_lim=4, y_lim=4, nx=25):
    # visualize 2D latent space
    # if args.latent_size == 2:
    ny = nx
    x_values = np.linspace(-x_lim, x_lim, nx)
    y_values = np.linspace(-y_lim, y_lim, ny)
    canvas = np.empty((args.input_size[1] * ny, args.input_size[2] * nx))

    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            zz = np.array([[xi], [yi]], dtype='float32').T
            z_mu = Variable(torch.from_numpy(zz).cuda(), volatile=True)
            x_mean, _ = model.p_x(z_mu)
            x = x_mean.data.cpu().numpy().flatten()

            canvas[(nx - i - 1) * args.input_size[1]:(nx - i) * args.input_size[1],
            j * args.input_size[2]:(j + 1) * args.input_size[2]] = x.reshape(args.input_size[1], args.input_size[2])

    fig = plt.figure(figsize=(8, 10))
    plt.imshow(canvas, origin="upper", cmap='Greys_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dir, 'latentSpace2D.png'), bbox_inches='tight')
    plt.close(fig)


# =======================================================================================================================
def plot_manifold2(model, args, dir, x_lim=4, y_lim=4, nx=25):
    # visualize 2D latent space
    # if args.latent_size == 2:
    ny = nx
    x_values = np.linspace(-x_lim, x_lim, nx)
    y_values = np.linspace(-y_lim, y_lim, ny)
    canvas = np.empty((args.input_size[1] * ny, args.input_size[2] * nx))

    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            zz2 = np.array([[xi], [yi]], dtype='float32').T
            z2_mu = Variable(torch.from_numpy(zz2).cuda(), volatile=True)
            z1_mu, _ = model.p_z1(z2_mu)
            x_mean, _ = model.p_x(z1_mu, z2_mu)
            x = x_mean.data.cpu().numpy().flatten()

            canvas[(nx - i - 1) * args.input_size[1]:(nx - i) * args.input_size[1],
            j * args.input_size[2]:(j + 1) * args.input_size[2]] = x.reshape(args.input_size[1], args.input_size[2])

    fig = plt.figure(figsize=(8, 10))
    plt.imshow(canvas, origin="upper", cmap='Greys_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dir, 'latentSpace2D.png'), bbox_inches='tight')
    plt.close(fig)
