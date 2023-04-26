# PyTorch implementation of [DCGAN](https://arxiv.org/pdf/1511.06434.pdf).
UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS.

## Modifications for Implementation
To ensure output size (64 x 64), I set kernel size, stride, padding size to 4, 2, 1, respectively.


## Run Code

```python
python train.py
```


## Generated Images
### MNIST


### CIFAR10


Epochs : 200

### LSUN


Epochs : 200

### IMAGENET (32 x 32 min-resized center crops)

Epochs : 200

<img src=>

## Walking in the latent space



### TODO
- [x] model code
- [x] train code
- [ ] add model save code
- [ ] add infer code : load check point
- [ ] add other datasets (LSUN, ImageNet)
- [ ] vis results
- [ ] add argparse for hyperparams
- [ ] update wandb to log losses 