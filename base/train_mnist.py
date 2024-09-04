import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from .networks import MNISTNet
from .datasets import RegressionMNIST


if __name__ == "__main__":

    # Transformation for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset
    ])


    # Download and load the MNIST training dataset
    train_dataset = RegressionMNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = RegressionMNIST(root='./data', train=False, download=True, transform=transform)

    # Define DataLoader parameters
    batch_size = 64
    shuffle = True

    # Download and load the MNIST validation dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=7)

    # Training parameters
    max_epochs = 100

    # Model
    model = MNISTNet()

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        dirpath='./model_checkpoints/mnist',
        filename='mnist-{epoch:02d}-{val_mae:.2f}',
        save_top_k=1,
        mode='min'
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback]
    )


    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)
