import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from base.networks import SubOpNet
from torch.utils.data import DataLoader
from base.datasets import SinFunction



if __name__ == "__main__":

    sin_function = SinFunction()

    # Create datasets
    num_samples_train = 100000
    num_samples_val = 2000
    num_samples_test = 10000

    train_dataset = SinFunction(-10,10).generate_dataset(num_samples_train)
    val_dataset = SinFunction(-10,10).generate_dataset(num_samples_val)
    test_dataset = SinFunction(-10,10).generate_dataset(num_samples_test)

    # Define DataLoader parameters
    batch_size = 64

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Usually, we do not shuffle the test set.

    # Training parameters
    max_epochs = 100

    # model
    model = SubOpNet()

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        dirpath='./checkpoints/base',
        filename='subopnet_sinfunction-{epoch:02d}-{val_mae:.2f}',
        save_top_k=1,
        mode='min'
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback]
    )

    # Train
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
