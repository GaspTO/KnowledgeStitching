from base.networks import StitchedOpNet, SubOpNet
from base.datasets import MathematicalFunction1
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

# Model
checkpoint_pretrained_frag = "checkpoints/experiments2/2_1/stitchednet_subopnet-epoch=45-val_mae=0.02.ckpt"
checkpoint_untrained_frag = "checkpoints/experiments2/2_1/untrainedfragment_stitchednet_subopnet-epoch=85-val_mae=0.02.ckpt"
checkpoint = checkpoint_untrained_frag

sub_op_net = SubOpNet()
model = StitchedOpNet(sub_op_net)
state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
model.load_state_dict(state_dict['state_dict'])


# Trainer
trainer = pl.Trainer()


# Test dataset
test_dataset = MathematicalFunction1(10,20).generate_dataset(10000)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Test stitched model
trainer.test(model, test_loader)


# Test model after resetting stitching layers
model.input_stitch.reset_parameters()
model.output_stitch.reset_parameters()
trainer.test(model, test_loader)
