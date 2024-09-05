It turns out there are already some works about this. This area seems to be named by most papers I've seen as *model stitching*, but one paper called it *model reassembly*.

We can divide this area into two questions:
* How to stitch different models
* What models to stitch

The works I've been seeing are focused on the first part.  The original idea of our work was based on the second part.  The idea was to stitch fragments specifically trained to recognize concepts given by our domain knowledge in an attempt to speed up and improve network training. So far, I haven't seen anyone do that.

## Stitching
The first paper I found was [StitchNet: Composing Neural Networks from Pre-Trained Fragments](https://arxiv.org/pdf/2301.01947), 

Others:
* [Stitchable Neural Networks](https://arxiv.org/pdf/2302.06586)
* [Deep Model Reassembly](https://arxiv.org/pdf/2210.17409)
 * [Revisiting Model Stitching to Compare Neural Representations](https://arxiv.org/abs/2106.07682)


## NET Similarity
A topic which seems frequent with stitching is the similarity between networks. Networks that have representation similarity seem to be more *stitchable* than others. Here's a bunch of works cited by the previous papers about that
* [Towards Understanding Learning Representations: To What Extent Do Different Neural Networks Learn the Same Representation](https://arxiv.org/abs/1810.11750)
* [Revisiting Model Stitching to Compare Neural Representations](https://arxiv.org/abs/2106.07682)
* [Similarity of Neural Network Representations Revisited](https://arxiv.org/pdf/1905.00414)