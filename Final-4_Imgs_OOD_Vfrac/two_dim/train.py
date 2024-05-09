import torch
import torch.nn as nn
from time import time
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from tqdm import tqdm
from two_dim.args import ArgParser
from utils.load_data import load_data
from models.generative_network import generative_network
from utils.plot import error_bar, plot_std, train_test_error
from utils.plot_samples import save_samples
from models.conditioning_network import conditioning_network


# this is the s and the t network
#def convolution_network(Hidden_layer):
#    return lambda input_channel, output_channel: nn.Sequential(
#        nn.Conv2d(input_channel, Hidden_layer, 3, padding=1),
#        nn.ReLU(),
#        nn.Conv2d(Hidden_layer, output_channel, 3, padding=1))


#def fully_connected(Hidden_layer):
#    return lambda input_data, output_data: nn.Sequential(
#        nn.Linear(input_data, Hidden_layer),
#        nn.ReLU(),
#        nn.Linear(Hidden_layer, output_data))

def convolution_network(Hidden_layer):
    return lambda input_channel, output_channel: nn.Sequential(
        nn.Conv2d(input_channel, Hidden_layer, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(Hidden_layer, output_channel, 3, padding=1))


def fully_connected(Hidden_layer):
    return lambda input_data, output_data: nn.Sequential(
        nn.Linear(input_data, Hidden_layer),
        nn.ReLU(),
        nn.Linear(Hidden_layer, output_data))



def init_networks(args, device, train_loader):
    x_shape = next(iter(train_loader))[0].shape
    y_shape = next(iter(train_loader))[1].shape
    
    cond_network = conditioning_network(y_shape).to(device)
    network_s_t = convolution_network(args.hidden_layer_channel)
    network_s_t2 = convolution_network(args.hidden_layer_channel2)
    network_s_t3 = fully_connected(args.hidden_layer3)
    INN_network = generative_network(args.cond_size, network_s_t,
                                     args.input_dimension1, args.input_dimension12, args.cond_size1, args.permute_a1,
                                     args.split_channel, args.input_dimension1_r,
                                     args.input_dimension2, args.input_dimension22, args.cond_size2, args.permute_a2,
                                     network_s_t2,
                                     args.input_dimension2_r,
                                     args.input_dimension3, args.input_dimension32, args.cond_size3, network_s_t3,
                                     args.permute_a3).to(device)

    combine_parameters = [parameters_net for parameters_net in INN_network.parameters() if parameters_net.requires_grad]
    for parameters_net in combine_parameters:
        parameters_net.data = 0.02 * torch.randn_like(parameters_net)

    combine_parameters += list(cond_network.parameters())
    optimizer = torch.optim.Adam(combine_parameters, lr=args.lr, weight_decay=args.weight_decay)

    generative_models_total_params = sum(p.numel() for p in INN_network.parameters()) / 1000000
    conditional_model_total_params = sum(p.numel() for p in cond_network.parameters()) / 1000000
    print(f"Params: gen network {generative_models_total_params}M, "
          f"cond network: {conditional_model_total_params}M")

    return INN_network, cond_network, device, optimizer


def train(epoch, loader, INN_network, cond_network, device, optimizer):
    number_of_batches = len(loader)
    #print('Train.py Train function')
    #print('number_of_train_batches: ', number_of_batches)
    INN_network.train()
    cond_network.train()
    loss_mean = []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x = x.view(16, 1, 64, 64)
        #print('enumerate x.shpape: -------------', x.shape) 
        if len(y.shape) != 4:
            y = y.view(16, 4, 64)
        #print('enumerate y.shpape: -------------', y.shape) 
        y1 = cond_network(y)
#         c = y1[2]
#         c2 = y1[1]
#         c3 = y1[0]
#         c4 = y1[3]
        
        c = y1[0]
        c2 = y1[1]
        c3 = y1[2]
        c4 = y1[3]
        
        z, log_j = INN_network(x, c, c2, c3, c4, forward=True)
        loss = torch.mean(z ** 2) / 2 - torch.mean(log_j) / (1 * 64 * 64)
        loss.backward()
        loss_mean.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        #if i % 9 == 0:
        if i == number_of_batches // 2 :
            print(f"Epoch {epoch} - {i}/{number_of_batches}: Train loss {loss:.3f}")
    loss_mean1 = loss_mean
    return loss_mean1


def test(epoch, loader, INN_network, cond_network, device):
    #print('Train.py Test function')
    INN_network.eval()
    cond_network.eval()
    loss_mean = []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x = x.view(16, 1, 64, 64)
        if len(y.shape) != 4:
            y = y.view(16, 4, 64)
        y1 = cond_network(y)
#         c = y1[2]
#         c2 = y1[1]
#         c3 = y1[0]
#         c4 = y1[3]

        c = y1[0]
        c2 = y1[1]
        c3 = y1[2]
        c4 = y1[3]
        z, log_j = INN_network(x, c, c2, c3, c4, forward=True)
        loss_val = torch.mean(z ** 2) / 2 - torch.mean(log_j) / (1 * 64 * 64)
        loss_mean.append(loss_val.item())
    loss_mean1 = loss_mean
    return loss_mean1


# def sample2(epoch, loader, INN_network, cond_network, device, args):
#     # print('Train.py sample2 function')
#     INN_network.eval()
#     cond_network.eval()
#     loss_mean = []
#     for batch_idx, (input, target) in enumerate(loader):
#         input, target = input.to(device), target.to(device)
#         x = input.view(1, 1, 64, 64),
#         labels_test = target
#         N_samples = 1000

#         labels_test = labels_test[0, :, :]
#         labels_test = labels_test.cpu().data.numpy()
#         l = np.repeat(np.array(labels_test)[np.newaxis, :, :], N_samples, axis=0)
#         l = torch.Tensor(l).to(device)
#         z = torch.randn(N_samples, 4096).to(device)
#         with torch.no_grad():
#             y1 = cond_network(l)
#             input = x.view(1, 4096)
#             c = y1[2]
#             c2 = y1[1]
#             c3 = y1[0]
#             c4 = y1[3]
#             val = INN_network(z, c, c2, c3, c4, forward=False)
#         rev_x = val.cpu().data.numpy()
#         if epoch % 10 == 0:
#             input_test = input[0, :].cpu().data.numpy()
#             input1 = input_test.reshape(1, 1, 64, 64)
#             samples1 = rev_x
#             samples12 = samples1
#             mean_samples1 = np.mean(samples1, axis=0)
#             mean_samples1 = mean_samples1.reshape(1, 1, 64, 64)
#             samples1 = samples1[:2, :, :, :]
#             x1 = np.concatenate((input1, mean_samples1, samples1), axis=0)
#             save_dir = '.'
#             save_samples(save_dir, x1, epoch, 2, 'sample', nrow=2, heatmap=True, cmap='jet')
#             std_sample = np.std(samples12, axis=0)
#             std_sample = std_sample.reshape(64, 64)

#             actual = input1
#             pred = rev_x
#             error_bar(actual, pred, epoch)
#             io.savemat('./results/samples_%d.mat' % epoch, dict([('rev_x_%d' % epoch, np.array(rev_x))]))
#             io.savemat('./results/input_%d.mat' % epoch, dict([('pos_test_%d' % epoch, np.array(input_test))]))
#         if epoch == (args.epochs - 1):
#             std_sample = np.std(rev_x, axis=0)
#             std_sample = std_sample.reshape(64, 64)
#             plot_std(std_sample, epoch)


# def test_NLL(epoch, loader, INN_network, cond_network, device, args):
#     # print('Train.py Test_NLL function')
#     INN_network.eval()
#     cond_network.eval()
#     final_concat = []
#     for batch_idx, (input, target) in enumerate(loader):
#         input, target = input.to(device), target.to(device)
#         input12, target = input.view(128, 1, 64, 64), target.view(128, 4,
#                                                                   64)  # for config_1  change this to target = target.view(128,2,64)
#         N_samples = 1000
#         labels_test1 = target

#         for jj in range(128):
#             labels_test = labels_test1[jj, :, :]
#             x = input12[jj, :, :, :]
#             labels_test = labels_test.cpu().data.numpy()
#             l = np.repeat(np.array(labels_test)[np.newaxis, :, :], N_samples, axis=0)
#             l = torch.Tensor(l).to(device)
#             z = torch.randn(N_samples, 4096).to(device)
#             with torch.no_grad():
#                 y1 = cond_network(l)
#                 input = x.view(1, 4096)
#                 c = y1[2]
#                 c2 = y1[1]
#                 c3 = y1[0]
#                 c4 = y1[3]
#                 val = INN_network(z, c, c2, c3, c4, forward=False)
#             rev_x = val.cpu().data.numpy()
#             input1 = x.cpu().data.numpy()
#             input1 = input1.reshape(1, 1, 64, 64)
#             rev_x = rev_x.reshape(1000, 1, 64, 64)

#             mean_val = rev_x.mean(axis=0)
#             mean_val = mean_val.reshape(1, 1, 64, 64)
#             d1 = (1 / domain) * np.sum(input1 ** 2)
#             n1 = (1 / domain) * np.sum((input1 - mean_val) ** 2)
#             m1 = n1 / d1
#             final_concat.append(m1)
#         final_concat = np.array(final_concat)
#     return final_concat
