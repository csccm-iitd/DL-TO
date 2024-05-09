import torch
import torch.nn as nn


class conditioning_network(nn.Module):
    '''conditioning network
        The input to the conditioning network are the observations (y)
        Args: 
        y: Observations (B X Obs)
    '''

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()

            def forward(self, x):
                return x.view(x.shape[0], -1)

        class Unflatten(nn.Module):
            def __init__(self, *args):
                super().__init__()

            def forward(self, x):
                if x[:, 0, 0].shape == (16,):
                    out = x.view(16, 4, 8, 8)  # for config_1  change this to out = x.view(16,2,8,8)
                elif x[:, 0, 0].shape == (1000,):
                    out = x.view(1000, 4, 8, 8)  # for config_1  change this to out = x.view(1000,2,8,8)
                elif x[:, 0, 0].shape == (1,):
                    out = x.view(1, 4, 8, 8)  # for config_1  change this to out = x.view(1,2,8,8)
                return out

        modules = []

        if self.input_shape[2:] == (64, 64):
            modules.append(nn.Sequential(nn.Conv2d(4,16,3,stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
#         elif self.input_shape[2:] == (32, 32):
#             modules.append(nn.Sequential(nn.Conv2d(4, 48, (3, 3), stride=2, padding=1),
#                                          nn.Conv2d(48, 48, (3, 3), stride=2, padding=2),
#                                          nn.ReLU(inplace=True),
#                                          nn.ConvTranspose2d(48, 48, 2, padding=1, stride=2)))
        else:
            print('Input Shape not Valid')
#             modules.append(nn.Sequential(Unflatten(),
#                                          nn.ConvTranspose2d(4, 48, 2, padding=0),
#                                          # for config_1  change this to nn.ConvTranspose2d(2,  48, 2, padding=0)
#                                          nn.ReLU(inplace=True),
#                                          nn.ConvTranspose2d(48, 48, 2, padding=1, stride=2)))

#         modules.extend([
#             nn.Sequential(nn.ReLU(inplace=True),
#                           nn.ConvTranspose2d(48, 96, 2, padding=0, stride=2),
#                           nn.ReLU(inplace=True),
#                           nn.ConvTranspose2d(96, 128, 3, padding=1, stride=1)),
#             nn.Sequential(nn.ReLU(inplace=True),
#                           nn.ConvTranspose2d(128, 128, 2, padding=0, stride=2)),
#             nn.Sequential(nn.ReLU(inplace=True),
#                           nn.AvgPool2d(6),
#                           Flatten(),
#                           nn.Linear(12800, 4800),
#                           nn.ReLU(inplace=True),
#                           nn.Linear(4800, 2048),
#                           nn.ReLU(inplace=True),
#                           nn.Linear(2048, 512))])
        modules.extend([
            nn.Sequential(nn.Conv2d(16,32,3,stride=2, padding=1),
                          nn.ReLU(inplace=True)),
            nn.Sequential(
                          nn.Flatten(),
                          nn.Linear(8192, 4096),
                          nn.ReLU(inplace=True),
                          nn.Linear(4096, 2048),
                          nn.ReLU(inplace=True),
                          nn.Linear(2048, 1024),
                          nn.ReLU(inplace=True),
                          nn.Linear(1024, 512))])

        self.multiscale = nn.ModuleList(modules)

    def forward(self, inp):
        module_outputs = [inp]
        for module in self.multiscale:

            module_input = module_outputs[-1]
            module_output = module(module_input)
            module_outputs.append(module_output)

        assert module_outputs[0].shape[1:] == (4, 64, 64)
        assert module_outputs[1].shape[1:] == (16, 32, 32)
        assert module_outputs[2].shape[1:] == (32, 16, 16)
        assert module_outputs[3].shape[1:] == torch.Size([512])
        return module_outputs[0:]
