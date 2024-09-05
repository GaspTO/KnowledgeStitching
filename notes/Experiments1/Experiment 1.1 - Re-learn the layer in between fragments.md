As planned, we resetted the last fully connected layer (the second one), and froze everything else.  Then we trained the ImageOpNet with the PairMNIST dataset.


# Results

#### 1. Compared the original ImageOpNet and the new one on the PairMNIST validationset

| Metric    | Base               | New               |
| --------- | ------------------ | ----------------- |
| Test Loss | 43.28394317626953  | 43.05772018432617 |
| Test MAE  | 1.8919484615325928 | 1.762453556060791 |

#### 2. Compared the original MNIST1 and the MNIST1 extracted from the new ImageOpNet on the MNIST validationset

| Metric    | Base Network        | New Network         |
| --------- | ------------------- | ------------------- |
| Test Loss | 0.16023200750350952 | 0.158871591091156   |
| Test MAE  | 0.10831695795059204 | 0.09800086915493011 |
A quick manual inspection also showed that it seems to have learned to decode numbers properly.


#### 3. Compared the ImageOpNet with reset on fc2 after only one training epoch

|              | Untrained         | After 1 Epoch       |
|--------------|-------------------|---------------------|
| Test Loss    | 3514.966796875    | 48.984657287597656  |
| Test MAE     | 44.74889373779297 | 3.03228759765625    |
	In fact, if we see the graph of the training we see that it the variability is bigger than the decrease:
![[Pasted image 20240905152642.png]]
Here's the same graph, but with maximum smoothing:
![[Pasted image 20240905153004.png]]


# Discussion
It is questionable whether or not we should aim for an equal score as the baseline, since the PairMNIST does not teach how to decode numbers - this is just a consequence. This task asks the MNISTnet to learn how to decode numbers as a requirement to solve PairMNIST.  

However, as we can see in the results, this wasn't the case -  we actually achieved a better score both in PairMNIST and MNIST. The score isn't insanely different, especially looking at the test loss, it is pretty much the same. So far, this makes us optimistic, since, as a consequence, the complex function signal of PairMNIST was able to reteach the last layer to learn MNIST. 
With this being said, this is not enough to prove that two networks can be stitched together since we knew it was possible in theory to stitch those two together again since this was its initial state.
I am however surprised that it wasn't harder to recover the language. 
