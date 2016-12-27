# Deconvolution-network-for-segmentation

A tensorflow implementation of [Learning Deconvolution Network for Semantic Segmentation](https://arxiv.org/pdf/1505.04366v1.pdf).</br>
A modified working version of [Fabian Bormann](https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation) implementation.</br></br>
Requires:</br>
 -Tensorflow</br>
 -numpy</br>
 -OpenCV2</br>

Requires the  PASCAL VOC 2012 dataset(stage-1 and stage-2) from the paper authors' website.Uses GPU for the convolution and deconvolution operations.A deconvolution network is learned separatelty for stage-1 and stage-2 training dataset and the final model is saved offline for later segmentation.
