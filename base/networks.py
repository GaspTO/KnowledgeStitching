import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam    
import pytorch_lightning as pl
import torchmetrics


class MNISTNet(pl.LightningModule):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(20)
        self.meanAbsoluteError = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.relu(x)
        x = self.max_pool2d(x) # global average pooling
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_mae', self.meanAbsoluteError(y_hat, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_mae', self.meanAbsoluteError(y_hat, y))
        return loss
    
    def save_model(self, file_path='mnist_net_checkpoint.ckpt'):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path='mnist_net_checkpoint.ckpt'):
        self.load_state_dict(torch.load(file_path, map_location=self.device))

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            else:
                print("Did not reset parameters for " + str(module))
    
    def __str__(self):
        return "MnistNet"


class OperatorNet(pl.LightningModule):
    def __init__(self):
        super(OperatorNet, self).__init__()
        self.fc1 = nn.Linear(2, 500)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 100)
        self.fc4 = nn.Linear(100, 1)
        self.meanAbsoluteError = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        return out

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_mae', self.meanAbsoluteError(y_hat, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_mae', self.meanAbsoluteError(y_hat, y))
        return loss

    def save_model(self, file_path='sum_net_checkpoint.ckpt'):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path='sum_net_checkpoint.ckpt'):
        self.load_state_dict(torch.load(file_path, map_location=self.device))

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            else:
                print("Did not reset parameters for " + str(module))

    def __str__(self):
        return "OpNet"


class ImageOperationNet(pl.LightningModule):
    def __init__(self, mnist_net1, mnist_net2, operation_net):
        super(ImageOperationNet, self).__init__()
        self.mnist_net1 = mnist_net1  # First MNISTNet instance
        self.mnist_net2 = mnist_net2  # Second MNISTNet instance
        self.operation_net = operation_net  # Operation network, can be any NN that takes two inputs
        self.meanAbsoluteError = torchmetrics.MeanAbsoluteError()

    def forward(self, image_pairs):
        image1 = image_pairs[:,0,:,:,:]
        image2 = image_pairs[:,1,:,:,:]
        num1 = self.mnist_net1(image1)
        num2 = self.mnist_net2(image2)
        combined_output = self.operation_net(torch.cat((num1, num2), dim=1))
        return combined_output

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx): #!
        image_pairs, target = batch
        target = target.view(-1,1)
        output = self.forward(image_pairs)
        loss = F.mse_loss(output, target)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx): #!
        image_pairs, target = batch
        target = target.view(-1,1)
        output = self.forward(image_pairs)
        loss = F.mse_loss(output, target)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        image_pairs, target = batch
        target = target.view(-1,1)
        output = self.forward(image_pairs)
        loss = F.mse_loss(output, target)
        self.log('test_loss', loss)
        self.log('test_mae', self.meanAbsoluteError(output, target))
        return loss

    def freeze_fragment(self, fragment):
        for param in fragment.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            else:
                print("Did not reset parameters for " + str(module))

    def save_model(self, file_path=None):
        if file_path is None:
            file_path = f'{str(self.operation_net)}_model.ckpt'  # Dynamic checkpoint name based on operation_net
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path=None):
        if file_path is None:
            file_path = f'{str(self.operation_net)}_model.ckpt'  # Ensures the correct file is loaded
        self.load_state_dict(torch.load(file_path, map_location=self.device))

    def __str__(self):
        return f'Image{str(self.operation_net)}'