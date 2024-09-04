import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from base.networks import OperatorNet
from torch.utils.data import DataLoader
from base.datasets import MathematicalFunction1



if __name__ == "__main__":

    mathematical_function_one = MathematicalFunction1()

    # Create datasets
    num_samples_train = 100000
    num_samples_val = 2000
    num_samples_test = 10000

    train_dataset = mathematical_function_one.generate_dataset(num_samples_train)
    val_dataset = mathematical_function_one.generate_dataset(num_samples_val)
    test_dataset = mathematical_function_one.generate_dataset(num_samples_test)

    # Define DataLoader parameters
    batch_size = 64
    shuffle = True

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Usually, we do not shuffle the test set.

    # Training parameters
    max_epochs = 25

    # model
    model = OperatorNet()

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        dirpath='./model_checkpoints/fun1',
        filename='fun1-{epoch:02d}-{val_mae:.2f}',
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
