from base.networks import MNISTNet, OperatorNet, ImageOperationNet
from base.datasets import PairMNIST, MathematicalFunction1, RegressionMNIST
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


# Model
mnist_checkpoint_path = "checkpoints/base/mnist-epoch=59-val_mae=0.11.ckpt"
op_checkpoint_path = "checkpoints/base/fun1-epoch=19-val_mae=0.00.ckpt"

mnist_net1 = MNISTNet.load_from_checkpoint(mnist_checkpoint_path)
mnist_net1.fc2.reset_parameters()

mnist_net2 = MNISTNet.load_from_checkpoint(mnist_checkpoint_path)
op_net = OperatorNet.load_from_checkpoint(op_checkpoint_path)
image_op_net = ImageOperationNet(mnist_net1, mnist_net2, op_net)

#checkpoint = "/home/to/Desktop/Tiago/Code/mycode/KnowledgeStitching/checkpoints/experiments1/image_op_net-epoch=94-val_mae=1.76.ckpt"
#image_op_net = ImageOperationNet(MNISTNet(), MNISTNet(), OperatorNet())
#state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
#image_op_net.load_state_dict(state_dict['state_dict'])

"""
Step 1.
Let's run ImageOpNet and MNISTnet after reseting fc2 parameters to 
make sure the error has increased.
The resuls are MAE 40.6 and 3.3, respectively.
"""
if False:
    # Trainer
    trainer = pl.Trainer()

    # Test ImageOpNet
    dataset = PairMNIST(MathematicalFunction1.apply, root="./data", train=False, download=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    trainer.test(image_op_net, loader)

    # Test MNISTnet
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset
    ])
    dataset = RegressionMNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    trainer.test(mnist_net1, loader)


"""
Step 2.
Train ImageOperationNet freezing everything except fc2
"""
max_epochs = 100

# Checkpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    monitor='val_mae',
    dirpath='checkpoints/experiments1/1_1',
    filename='image_op_net-{epoch:02d}-{val_mae:.2f}',
    save_top_k=1,
    mode='min'
)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=max_epochs,
    callbacks=[checkpoint_callback]
)

i = 0
for name, param in image_op_net.named_parameters():
    if 'mnist_net1.fc2' not in name:
        param.requires_grad = False
    else:
        i += 1
assert i == 2

train_dataset = PairMNIST(MathematicalFunction1.apply, root="./data", train=True)
val_dataset = PairMNIST(MathematicalFunction1.apply, root="./data", train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6)

trainer.test(image_op_net, val_loader) 
trainer.fit(image_op_net, train_loader, val_loader) 