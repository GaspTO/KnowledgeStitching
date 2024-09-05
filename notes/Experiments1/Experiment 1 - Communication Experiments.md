
The foundation of stitching is that one network (or part of) can be connected to a base network. When two networks are trained separately, they use a different *language*. The embedding space they use in the different layers is different from one another. Therefore, for stitching to happen, there needs to be a way for both networks to be able to communicate. 

In this set of experiments, we attempt to understand if such is possible and in which conditions.  For that, we create a network called ImageOperationNet, which receives two MNIST (number) images and applies a complex mathematical function to them, returning 
one float value.  The ImageOperationNet is constructed by different fragments. We first apply a MNISTnet to each input separately, which outputs a float value corresponding to the number in the image. Second, we run an OperatorNet to the concatenated output of both MNISTnets which applies some mathematical complex operation which returns one float value.  


## Experiment 1 - Can one fragment learn the language of another?

The fundamental question we are going to test in this subset of experiments is whether or not one fragment can learn the specific language of another unilaterally (i.e. without changing the second fragment's language).  

We will start with a control experiment, where each fragment has been trained separately and simply connected/stitched to each other as previously described.  All the following experiments attempt to reset and re-train parameters of first MNISTnet to check whether it can re-learn the language of the second fragment, which it is the one it knows before any parameters are changed. 

For details on the results, see [[Experiment 1 - Control]] - Commit #2

### Experiment 1.1.
#### What
Reset the parameters of the last linear layer of the first MNISTnet. Then train the ImageOperationNet, unfreezing only the linear layer's parameters. 
#### Why
The idea here is to see if a simple linear layer can recover the number language. In theory, since it has learned it before, it should be able to learn it again, but as is experience, neural networks seem to need more parameters than what are strictly necessary in order to learn successfully.
#### Results
For details on the results, see [[Experiment 1.1 - Re-learn the layer in between fragments]]


### Experiment 1.1.A - not done
#### What
If the previous experiment was unsuccessful,  add a couple linear layers between the fragments. 
#### Why
The idea here is to see if, to learn how to communicate with a specific fragment, we do need more than the strictly necessary number of parameters to do so. Following the rationale of the previous experiment.
#### Results
This experience was not conducted since the experiment 1.1 was successful.



### Experiment 1.1.B - not done
#### What
If the previous experiment was unsuccessful, go back to the control network and reset the first MNISTnet, then train just this network in the context of the ImageOperationNet.
#### Why
This is a more drastic experiment. If every other experiment failed except this, then the conclusion is that it is not possible for a network to learn the language of another unilaterally, except if it trained from scratch in the context of the stitch. This is still interesting in the sense that, we might be able to augment networks that are already trained, but it complicates the job of adding already trained fragments.
#### Results
This experience was not conducted since the experiment 1.2 was not done



### Experiment 1.2.
#### What
In the successful experiment 1.1., we resetted only the last fully connected layer, but there are actually two of them. We will reset both now and see what happens.
#### Why
The idea is to distance ourselves even more between the last unresetted embeddings of MNISTnet1 and the OPNet.