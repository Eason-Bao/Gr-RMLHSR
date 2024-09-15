# Gr-RMLHSR: Learning a Discriminative Grassmannian Neural Network for Visual Classification
## Contribution
>
• A Riemannian metric learning regularization module for improving the model capacity is proposed.<br />
• A hard sample reward strategy for refining the discriminability of the learning features is suggested.
> 
## Network
![](https://github.com/Eason-Bao/Gr-RMLHSR/blob/main/Network.png)

## Usage
The [AFEW](https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/AFEW_SPD_data.zip) dataset is publicly available.<br />
If you want to train and evaluate the original Grassmann network, please run demo.py.<br />
If you want to train and evaluate the Grassmann network with the PM metric learning term, please run demo_PM.py.<br />
If you want to train and evaluate the Grassmann network with the PAM metric learning term, please run demo_PAM.py.<br />
If you want to train and evaluate the Grassmann network with a hard sample reward strategy, please run demo_PAM_HSR.py.
