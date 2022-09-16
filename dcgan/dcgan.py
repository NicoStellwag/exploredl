## imports

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob

##
## global objects

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(f"./dcgan/runs/{timestamp}")

##
## global params

train = True
workers = 2 # data preprocessing uses 2 subprocesses
num_gpu = 0

batch_size = 128
sigma_init = 0.02 # init all weights with zero centered normal distribution
slope_lrelu = 0.2 # slope of leaky relu in all models
lr = 2e-4
beta1_adam = 0.5
nz = 100 # size of the noise vector
image_size = 64
epochs = 5

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

# example_batch = next(iter(dataloader))
# plt.figure(figsize=(3, 3))
# plt.axis('off')
# plt.title('train images')
# plt.imshow(vutils.make_grid(example_batch[0].to(device)[:9], padding=2, normalize=True, nrow=3).permute(1, 2, 0))
# plt.show()

##
## models

def init_weights(m):
    layername = type(m).__name__
    if layername.find("Conv") >= 0:
        nn.init.normal_(m.weight.data, 0.0, sigma_init)
    elif layername.find("BatchNorm") >= 0:
        nn.init.normal_(m.weight.data, 1.0, sigma_init)
        nn.init.constant_(m.bias.data, 0.0)

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
            nn.BatchNorm2d(256),
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

dis = Discriminator().to(device)
gen = Generator().to(device)
if device.type == 'cuda' and num_gpu > 1:
    dis = nn.parallel.DataParallel(dis, list(range(num_gpu)))
    gen = nn.parallel.DataParallel(gen, list(range(num_gpu)))
dis.apply(init_weights)
gen.apply(init_weights)

##
## loss and optimizer

# discriminator:
#       max log(D(x)) + log(1 - D(G(x)))
# <=>   min -log(D(x)) + -log(1-D(G(x)))
# =>    D(x) should be 1 and D(G(x)) should be 0
y_dis_real = torch.ones((batch_size, 1), dtype=torch.float, device=device)
y_dis_fake = torch.zeros((batch_size, 1), dtype=torch.float, device=device)
# generator:
#       max log(D(G(x)))
# <=>   min -log(D(G(x)))
# =>    D(G(x)) should be 1
y_gen_fake = y_dis_real

criterion = nn.BCELoss()
optimizer_dis = torch.optim.Adam(dis.parameters(), lr=lr, betas=(beta1_adam, 0.999))
optimizer_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1_adam, 0.999))

##
## training loop

path_dis = f'./dcgan/model/dis_{timestamp}.pth'
path_gen = f'./dcgan/model/gen{timestamp}.pth'

z_fixed = torch.randn((9, nz, 1, 1), device=device) # fixed noise to track the gen's progress
batches = len(dataloader)
running_loss_dis = 0
running_loss_gen = 0

if train:
    for epoch in range(epochs):
        for i, (images_real, _) in enumerate(dataloader):
            # TODO make sure the label vectors are of same size as the output vectors for the last batch of the epoch
            # TODO why the hell is the dis loss 0 after 1 or 2 epochs

            # real data prep
            X_real = images_real.to(device)

            # fake data prep
            z = torch.randn((batch_size, nz, 1, 1), device=device)
            X_fake = gen(z)

            # discriminator forward real
            out = dis(X_real).view(-1, 1)
            loss_dis_real = criterion(out, y_dis_real)

            # discriminator backward real
            optimizer_dis.zero_grad()
            loss_dis_real.backward()

            # discriminator forward fake
            out = dis(X_fake.detach()).view(-1, 1) # detach the fakes because we will recalc later
            loss_dis_fake = criterion(out, y_dis_fake)

            # discriminator backward fake
            loss_dis_fake.backward()
            optimizer_dis.step()
            loss_dis = loss_dis_real + loss_dis_fake # only relevant for tracking progress

            # generator forward
            out = dis(X_fake).view(-1, 1) # recalculate to optimize against the updated discriminator
            loss_gen = criterion(out, y_gen_fake)

            # generator backward
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # logging
            running_loss_dis += loss_dis
            running_loss_gen += loss_gen
            if (i + 1) % 50 == 0:
                print(f"epoch {epoch+1}/{epochs} - batch {i+1}/{batches} - dis loss {loss_dis:.5f} - gen loss {loss_gen:.5f}")
                writer.add_scalar('running loss Discriminator', running_loss_dis / 50, epoch * batches + i)
                writer.add_scalar('running loss Generator', running_loss_gen / 50, epoch * batches + i)
                running_loss_dis, running_loss_gen = 0, 0

            # track gen's progress using fixed noise
            if (i % 500 == 0) or ((epoch == epochs - 1) and (i == batches - 1)):
                with torch.no_grad():
                    fakes = gen(z_fixed).detach().cpu()
                    plt.figure(figsize=(3, 3))
                    plt.axis('off')
                    plt.title('Fixed noise images')
                    plt.imshow(vutils.make_grid(fakes, padding=2, normalize=True, nrow=3).permute(1, 2, 0))
                    writer.add_figure('fixed noise gen results', plt.gcf())
                    plt.clf()

    torch.save(dis.state_dict(), path_dis)
    torch.save(gen.state_dict(), path_gen)

##
## validate

if not train:
    saved_dis_models = glob.glob('./dcgan/model/dis*.pth')
    saved_gen_models = glob.glob('./dcgan/model/gen*.pth')
    latest_dis_model = max(saved_dis_models, key=os.path.getctime)
    latest_gen_model = max(saved_gen_models, key=os.path.getctime)
    gen, dis = Generator(), Discriminator()
    dis.load_state_dict(torch.load(path_dis))
    gen.load_state_dict(torch.load(path_gen))
    dis.eval()
    gen.eval()

    with torch.no_grad():
        fakes = gen(z_fixed).detach().cpu()
        plt.figure(figsize=(3, 3))
        plt.axis('off')
        plt.title('Fixed noise images')
        plt.imshow(vutils.make_grid(fakes, padding=2, normalize=True, nrow=3).permute(1, 2, 0))
        writer.add_figure('fixed noise gen results', plt.gcf())
        plt.clf()

##
