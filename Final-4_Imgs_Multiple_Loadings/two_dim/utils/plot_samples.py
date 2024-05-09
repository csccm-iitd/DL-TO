import os
import pathlib
import shutil

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
import numpy as np
import torchvision.utils

plt.switch_backend('agg')
import scipy.io as io


def save_samples(save_dir, images, epoch, plot, name,  nrow=4, heatmap=True, cmap='jet', target_images=None):
    """Save samples in grid as images or plots
    Args:
        images (Tensor): B x C x H x W
    """
    vmin1 = [np.amin(images[0, :, :, :])]
    vmax1 = [np.amax(images[0, :, :, :])]
    if images.shape[0] < 10:
        nrow = 2
        ncol = images.shape[0] // nrow
    else:
        ncol = nrow

    epoch_dir = save_dir + f'/{name}_epoch{epoch}'
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)

    if heatmap:
        for c in range(images.shape[1]):
            # (11, 12)
            fig = plt.figure(1, (12, 12))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(nrow, ncol),
                             axes_pad=0.3,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="3%",
                             cbar_pad=0.1
                             )
            for j, ax in enumerate(grid):
                im = ax.imshow(images[j][c], cmap='jet', origin='lower',
                               interpolation='bilinear', vmin=vmin1[j % 1], vmax=vmax1[j % 1])
                if j == 0:
                    ax.set_title('actual')
                elif j == 1:
                    ax.set_title('mean')
                else:
                    ax.set_title('sample %d' % (j - 1))
                ax.set_axis_off()
                ax.set_aspect('equal')
            cbar = grid.cbar_axes[0].colorbar(im)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.toggle_label(True)
            plt.subplots_adjust(top=0.95)
            plt.savefig(epoch_dir + '/output{}.pdf'.format(c),
                        bbox_inches='tight')
            #plt.close(fig)
    else:
        torchvision.utils.save_image(images,
                                     epoch_dir + '/fake_samples.png',
                                     nrow=nrow,
                                     normalize=True)

    if target_images is not None:
        vmin1 = [np.amin(target_images[:, :, :])]
        vmax1 = [np.amax(target_images[:, :, :])]
        nrow = 4
        ncol = 1
        fig = plt.figure(1, (12, 12))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(nrow, ncol),
                         axes_pad=0.3,
                         share_all=False,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="3%",
                         cbar_pad=0.1
                         )
        for j, ax in enumerate(grid):
            im = ax.imshow(target_images[j], cmap='jet', origin='lower',
                           interpolation='bilinear', vmin=vmin1[j % 1], vmax=vmax1[j % 1])

            ax.set_title('sample %d' % (j))
            ax.set_axis_off()
            ax.set_aspect('equal')
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.toggle_label(True)
        plt.subplots_adjust(top=0.95)
        plt.savefig(epoch_dir + '/y.pdf',
                    bbox_inches='tight')
       # plt.close(fig)
