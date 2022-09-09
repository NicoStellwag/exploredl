## imports

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

##
## global objects

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##
## global params

workers = 2 # data preprocessing uses 2 subprocesses
num_gpu = 0

batch_size = 128
sigma_init = 0.02 # init all weights with zero centered normal distribution
slope_lrelu = 0.2 # slope of leaky relu in all models
lr = 2e-4
beta_adam = 0.5
nz = 100 # size of the noise vector
image_size = 64

##
## data

dataset = dset.ImageFolder('./dcgan/data',
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size, True, num_workers=workers)

example_batch = next(iter(dataloader))
plt.figure(figsize=(3, 3))
plt.axis('off')
plt.title('train images')
plt.imshow(vutils.make_grid(example_batch[0].to(device)[:9], padding=2, normalize=True, nrow=3).permute(1, 2, 0))
plt.show()


##
## models

def init_weights(m):
    layername = type(m).__name__
    if layername.find("Conv") >= 0:
        nn.init.normal_(m.weight.data, 0.0, sigma_init)
    elif layername.find("BatchNorm") >= 0:
        nn.init.normal_(m.weight.data, 1.0, sigma_init)
        nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, 1024, 4, 1, 0, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                # -> n, 1024, 4, 4
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # -> n, 512, 8, 8
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # -> n, 256, 16, 16
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # -> n, 128, 32, 32
                nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
                nn.Tanh(),
                # -> n, 3, 64, 64
                )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                # n, 3, 64, 64
                nn.Conv2d(3, 128, 4, 2, 1, bias=False),
                nn.LeakyReLU(slope_lrelu, inplace=True),
                # -> n, 128, 32, 32
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(slope_lrelu, inplace=True),
                # -> n, 256, 16, 16
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(slope_lrelu, inplace=True),
                # -> n, 512, 8, 8
                nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(slope_lrelu, inplace=True),
                # -> n, 1024, 4, 4
                nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                # -> n, 1
                )

    def forward(self, input):
        return self.main(input)

gen = Generator().to(device)
dis = Discriminator().to(device)
if device.type == 'cuda' and num_gpu > 1:
    gen = nn.parallel.DataParallel(gen, list(range(num_gpu)))
    dis = nn.parallel.DataParallel(dis, list(range(num_gpu)))
gen.apply(init_weights)
dis.apply(init_weights)

##
# continue: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#loss-functions-and-optimizers
