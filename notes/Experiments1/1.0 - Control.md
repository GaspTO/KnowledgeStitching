We have achieved a 0.6 MAE result on the ImageOperationNet after the MNIST and OperationNet have been trained.  This has been all saved in the second commit.  We used L4 GPU in the LightningStudio to achieve this.

For the MNISTnet we achieved MAE=0.11
and for the complex function we achieved MAE=0.03 (although the test data was different from the one used to test during training, since we generate the data on the fly. This last one had MAE=0.00).

I believe we are ready to proceed with the rest of the experiments.
