There are two questions which require assessing if knowledge stitching is to work. The first is whether or not networks can be stitched - in other words, if they can learn to communicate with one another. What an embedding vector means to one fragment means something completely different to another. If they are to communicate, they need a bridge in between to translate from one's embeddings, to another's. The second question is whether a network is going to learn to use the stitched network during its training, or is going to ignore it.

In this set of experiments, we focus on the first question. We create a simple network that is formed by stitching three networks together that have been trained separately. The idea here is to reset some parameters in the layers that border these fragments and see if they can be relearned.

In specific, we create a network called ImageOperationNet, which receives two MNIST (number) images and applies a complex mathematical function to them, returning 
one float value.  The ImageOperationNet is constructed by different fragments. We first apply a MNISTnet to each input separately, which outputs a float value corresponding to the number in the image. Second, we run an OperatorNet to the concatenated output of both MNISTnets which applies some mathematical complex operation which returns one float value.   For more detail, read  [[1.0 - Control]].


![[ImageOpNet.png]]


### Experiment 1.1. - Reset last fc layer of MNISTnet1
see [[1.1 - Reset last fc layer of MNISTnet1]]

commit f03426dc5af5d7f696b0c3098ea7f09a8c1fc450 (HEAD -> main, origin/main)
Author: Tiago Oliveira <tiagojgroliveira@tecnico.ulisboa.pt>
Date:   Thu Sep 5 16:46:17 2024 +0100

    Milestone: Finish Experiment 1.1.

### Experiment 1.2. -  Reset all (two) fc layer of MNISTnet1
see [[1.2. - Reset all (two) fc layer of MNISTnet1]]

