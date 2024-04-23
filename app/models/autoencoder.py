import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from app.common.constants import DEVICE, DEFAULT_BATCH_SIZE


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=(1, 1), is_last=False):
        super().__init__()
        self.in_channels, self.out_channels = (in_channels, out_channels)
        self.kernel_size = (kernel_size,)
        self.padding = (padding,)
        self.stride = stride
        self.is_last = is_last
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size[0], padding=self.padding, stride=self.stride)
        self.normalization = nn.BatchNorm2d(num_features=self.out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if not self.is_last:
            x = self.normalization(x)
            x = self.act(x)
        return x


class EncodeBlock(ConvBlock):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, is_last=False):
        super().__init__(in_channels, out_channels, kernel_size, padding, stride, is_last)
        self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = super().forward(x)
        x = self.max_pooling(x)
        return x


class DecodeBlock(ConvBlock):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, scale_factor=2,
                 is_last=False):
        super().__init__(in_channels, out_channels, kernel_size, padding, stride, is_last)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', recompute_scale_factor=False)
        x = super().forward(x)
        return x


class AutoencoderModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_channels = 8
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.encoder = nn.Sequential(
            ConvBlock(3, self.base_channels, kernel_size=(5, 5), padding=2),
            ConvBlock(self.base_channels, self.base_channels, kernel_size=(5, 5), padding=2),
            EncodeBlock(in_channels=self.base_channels, out_channels=self.base_channels),
            ConvBlock(self.base_channels, self.base_channels * 2),
            ConvBlock(self.base_channels * 2, self.base_channels * 2),
            EncodeBlock(in_channels=self.base_channels * 2, out_channels=self.base_channels * 2, is_last=True))
        self.decoder = nn.Sequential(
            DecodeBlock(in_channels=self.base_channels * 2, out_channels=self.base_channels),
            DecodeBlock(in_channels=self.base_channels, out_channels=3, is_last=True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.sigmoid(x)
        return x

    @torch.inference_mode()
    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return x


def train(model, data_loader, optimizer, loss_fn):
    model.to(DEVICE)
    model.train()

    train_loss = []
    for x, _ in data_loader:
        x = x.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, x)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def fine_tune(model, train_data):
    loss_fn = nn.MSELoss()
    optimizer = Adam([
        {'params': model.encoder[2].parameters(), 'lr': 1e-4},
        {'params': model.encoder[3].parameters(), 'lr': 1e-4},
        {'params': model.encoder[4].parameters(), 'lr': 1e-3},
        {'params': model.encoder[5].parameters(), 'lr': 1e-3},
        {'params': model.decoder[0].parameters(), 'lr': 1e-3},
        {'params': model.decoder[1].parameters(), 'lr': 1e-4}
    ])
    data_loader = DataLoader(dataset=train_data, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    for epoch in range(1, 11):
        train(model, data_loader, optimizer, loss_fn)


@torch.inference_mode()
def get_embeddings(model, data):
    model.to(DEVICE)
    model.eval()

    result = []
    for x, _ in DataLoader(data, batch_size=DEFAULT_BATCH_SIZE):
        x = x.to(DEVICE)
        out = model.encode(x)
        result.extend(out.cpu())

    return torch.stack(result)


def get_transformer():
    return T.Compose([
        T.Resize((192, 192)),
        T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
    ])
