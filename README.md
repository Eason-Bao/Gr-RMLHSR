# Gr-RMLHSR
## Abstract
> Learning representations on the Grassmannian manifolds is popular in quite a few visual classification tasks. With the development of deep learning techniques, several neural networks have recently emerged for processing subspace data. However,  the diversely changed appearance of the signal data (e.g., video clips and image sets),  makes it impossible for the existing Grassmannian networks (GrasNets) that rely on a single cross-entropy loss for end-to-end training to learn effective geometric representations, especially for complicated data scenarios. To solve this problem, a Riemannian triplet loss-based Riemannian metric learning mechanism is introduced to the original GrasNet, which can explicitly encode and learn the characteristics of the intra- and inter-class data distributions conveyed by the input data during network training. Additionally, given the existence of intra-class diversity and inter-class ambiguity of the input data, we propose a hard sample reward strategy (HSR) to further improve the discriminability of the learned network embedding. Extensive experimental results obtained on four benchmarking datasets demonstrate the effectiveness of the proposed method.
> 
## Network
![](https://github.com/Eason-Bao/Gr-RMLHSR/blob/main/Network.png)

## Usage
Run demo.py to train and evaluate the network.
