# cvae
This is an implementation of conditional variational autoencoders inspired by the paper [Learning Structured Output Representation using Deep Conditional Generative Models](http://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) by K. Sohn, H. Lee, and X. Yan.

The formatting of this code draws its initial influence from [Joost van Amersfoort's](https://github.com/y0ast/VAE-Torch) implementation of Kingma's variational autoencoder. 

In addition to the vanilla formulation of the VAE (which uses a diagonal covariance gaussian distribution as its prior), I have also introduced the use of mixture of gaussians as prior. This increases significantly increases the generative model's ability to tackle data that is highly structured and whose distribution is multi-modal. 

As of the moment, the use of GPU is hard-coded in some parts of the code. But this will be fixed later.