from base.networks import MNISTNet, OperatorNet, ImageOperationNet
from base.datasets import PairMNIST, MathematicalFunction1, RegressionMNIST
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Model
mnist_checkpoint_path = "checkpoints/base/mnist-epoch=59-val_mae=0.11.ckpt"
op_checkpoint_path = "checkpoints/base/fun1-epoch=19-val_mae=0.00.ckpt"

mnist_net1 = MNISTNet.load_from_checkpoint(mnist_checkpoint_path)
mnist_net2 = MNISTNet.load_from_checkpoint(mnist_checkpoint_path)
op_net = OperatorNet.load_from_checkpoint(op_checkpoint_path)
image_op_net = ImageOperationNet(mnist_net1, mnist_net2, op_net)


# Trainer
trainer = pl.Trainer()


# Test ImageOpNet
dataset = PairMNIST(MathematicalFunction1.apply, root="./data", train=False)
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


# Opnet
dataset = MathematicalFunction1().generate_dataset(2000)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
trainer.test(op_net, loader)
