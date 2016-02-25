# cvae
This is an implementation of conditional variational autoencoders inspired by the paper [Learning Structured Output Representation using Deep Conditional Generative Models](http://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) by K. Sohn, H. Lee, and X. Yan.

The formatting of this code draws heavy influence from [Joost van Amersfoort's](https://github.com/y0ast/VAE-Torch) implementation of Kingma's variational autoencoder. 

Test out the implementation with either the MNIST or (my made-up) Flipshift dataset
```
th main.lua -dataset mnist -gpu 1
th main.lua -dataset flipshift -gpu 1
```

Compare CVAE with a vanilla MLP
```
th main_mlp.lua -dataset mnist -gpu 1
th main_mlp.lua -dataset flipshift -gpu 1
```

Also check out the visualization in the itorch notebook: `visualization.ipynb`





