## imports and device

import random
from datetime import datetime
import torch
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(f"./multiclass_img/runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

##
## hyperparameters and other definitions

lr = 1e-4
epochs = 15
batch_size=200

classnames = {
    0: 'TShirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'AnkleBoot',
}

##
## data

data_train  = torchvision.datasets.FashionMNIST('./multiclass_img/data/train', True, download=True, transform=transforms.ToTensor())
data_val = torchvision.datasets.FashionMNIST('./multiclass_img/data/val', False, download=True, transform=transforms.ToTensor())
# print(len(data_train))

loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
loader_val = torch.utils.data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=False)

examples = iter(loader_val)
ex_img, ex_label = examples.next()
print(ex_img.shape, ex_label.shape)
print(ex_label)
# plt.imshow(ex_img[0, 0], cmap='gray')
# plt.show()

##
## model

class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 6, 3)
        self.fc1 = nn.Linear(6 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # -> n, 2, 13, 13
        x = self.pool(F.relu(self.conv2(x))) # -> n, 6, 5, 5
        x = x.view(-1, 6 * 5* 5) # -> n, 6 * 5 * 5
        x = F.relu(self.fc1(x)) # -> n, 64 
        x = self.fc2(x) # -> n, 10
        return x

model = FashionModel().to(device)

##
## loss and optimizer

criterion = nn.CrossEntropyLoss() # expects unnormalized input, hence no activation function in output layer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

##
## train and safe

path_model = './multiclass_img/model/fashion.pth'
running_loss = 0

train = False
if train:
    batches = len(loader_train)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(loader_train):
            # push data on gpu
            X = images.to(device)
            y = labels.to(device)

            # forward
            y_pred = model(X)
            loss = criterion(y_pred, y)

            # backward and optimize
            optimizer.zero_grad() # grads of parameters will accumulate for the passes
            loss.backward() # computes grads of all tensors with requires_grad=True (set for parameters by model)
            optimizer.step() # adjusts the parameters according to their grads
            running_loss += loss.item()

            # logging
            if (i+1) % 50 == 0:
                print(f"epoch {epoch+1}/{epochs}, batch {i+1}/{batches}, loss {loss.item():.5f}")
                writer.add_scalar('running loss', running_loss / 50, epoch * len(loader_train) + i)
                running_loss = 0

    torch.save(model.state_dict(), path_model)

##
## evaluate

model = FashionModel()
model.load_state_dict(torch.load(path_model)) # the correct device should be saved with the model, otherwise load gets the map_location argument
model.eval()
with torch.no_grad():
    total_corr = 0
    total_samples = 0
    byclass_corr = [0] * 10
    byclass_samples = [0] * 10
    for images, labels in loader_val:
        # push data on gpu
        X = images.to(device)
        y = labels.to(device)

        # forward
        y_pred = model(X)

        # calc statistics
        total_samples += y.shape[0]
        _, y_pred_scalar = torch.max(y_pred, 1) # max of dim 1, returns (val, index): (tensor, tensor)
        total_corr += (y == y_pred_scalar).sum().item()
        for i in range(batch_size):
            if y[i] == y_pred_scalar[i]:
                byclass_corr[y[i]] += 1
            byclass_samples[y[i]] += 1

accuracy = (total_corr / total_samples) * 100.0
print(f"accuracy: {accuracy}")

##
## send some predictions to tensorboard
imgs, lbls = iter(loader_val).next()
out = model(imgs.to(device))
fig = plt.figure()
_, axs = plt.subplots(3, 3)
for i in range(9):
    index = random.randrange(0, imgs.shape[0])
    _, out_scalar = torch.max(out[index], 0)
    plot_label = f"{classnames[lbls[index].item()]}: {classnames[out_scalar.item()]}"
    axs.flat[i].imshow(imgs[index, 0], cmap='gray')
    axs.flat[i].set_title(plot_label)
writer.add_figure('some predictions', plt.gcf())
# plt.show()

##
