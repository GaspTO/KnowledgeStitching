from base.networks import SubOpNet, StitchedOpNet
from base.datasets import PairMNIST, MathematicalFunction1, RegressionMNIST
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
import torch



## Instantiate subopnet
PRETRAINED=True
if PRETRAINED:
    sub_op_net = SubOpNet.load_from_checkpoint("checkpoints/base/subopnet_sinfunction-epoch=98-val_mae=0.01.ckpt")
    for param in sub_op_net.parameters():
        param.requires_grad = False
else:
    sub_op_net = SubOpNet()

## Instantite stitchednet
stitched_net = StitchedOpNet(sub_op_net=sub_op_net)



# Create datasets
num_samples_train = 100000
num_samples_val = 2000
num_samples_test = 10000

##! change the range in the validation set
train_dataset = MathematicalFunction1().generate_dataset(num_samples_train)
val_dataset = MathematicalFunction1(-10,20).generate_dataset(num_samples_val)
test_dataset = MathematicalFunction1(-10,20).generate_dataset(num_samples_test)

# Define DataLoader parameters
batch_size = 32

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Usually, we do not shuffle the test set.

# Training parameters
max_epochs = 100


# Checkpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    monitor='val_mae',
    dirpath='./checkpoints/experiments2/2_1',
    filename='pretrainedfragment_stitchednet_subopnet-{epoch:02d}-{val_mae:.2f}',
    save_top_k=1,
    mode='min'
)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=max_epochs,
    callbacks=[checkpoint_callback]
)

# Train
trainer.fit(stitched_net, train_loader, val_loader)
trainer.test(stitched_net, test_loader)


