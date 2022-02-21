# Modified from
# https://github.com/yjn870/SRCNN-pytorch/blob/064dbaac09859f5fa1b35608ab90145e2d60828b/models.py
# https://github.com/yjn870/SRCNN-pytorch/blob/064dbaac09859f5fa1b35608ab90145e2d60828b/train.py


import copy

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

from tqdm import tqdm


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64,
                               kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32,
                               kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels,
                               kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def train_srcnn(train_h5_file, lr=1e-4, num_epochs=400,
                batch_size=16, num_workers=8):
    """
    Very basic training loop for SRCNN. Validation and model checkpoint saving are omitted for simplicity.
    
    Args:
        train_h5_file: HDF5 file containing the training images. You can download this from
            https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0
        lr: The learning rate used for the Adam optimizer.
        num_epochs: The number of epochs.
        batch_size: How many images to put into a single training batch.
        num_workers: The number of CPU-workers to use for data loading (= composing the batch).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(),
         'lr': lr * 0.1}
    ], lr=lr)

    train_dataset = TrainDataset(train_h5_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    model.train()

    for epoch in tqdm(range(num_epochs)):
        loss_sum = 0
        step_counter = 0
        
        for data in train_dataloader:
            lr_imgs, hr_imgs = data

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            pred_hr_imgs = model(lr_imgs)

            loss = criterion(pred_hr_imgs, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss
            step_counter += 1

        loss_avg = loss_sum/step_counter

        print(f'Avg train loss: {loss_avg} at epoch {epoch}/{num_epochs}')


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])