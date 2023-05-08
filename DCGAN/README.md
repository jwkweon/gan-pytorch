# PyTorch implementation of [DCGAN](https://arxiv.org/pdf/1511.06434.pdf).
UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS.

## Modifications for Implementation
To ensure output size (64 x 64), I set kernel size, stride, padding size to 4, 2, 1, respectively.


## Run Code

```ShellSession
$ python train.py
```


## Generated Images
### CIFAR10

<img src=results\cifar10\0010_results_cifar10.png>
<img src=results\cifar10\0020_results_cifar10.png>
<img src=results\cifar10\0030_results_cifar10.png>
<img src=results\cifar10\0040_results_cifar10.png>
<img src=results\cifar10\0050_results_cifar10.png>

Epochs : 10 - 50


<img src=results\cifar10\0100_results_cifar10.png>
<img src=results\cifar10\0200_results_cifar10.png>
<img src=results\cifar10\0300_results_cifar10.png>
<img src=results\cifar10\0500_results_cifar10.png>
<img src=results\cifar10\1000_results_cifar10.png>

Epochs : 100, 200, 300, 500, 1000

<img src=results\cifar10\1000_gif_results_cifar10.gif>

Epochs : 1000

### LSUN


Epochs : 200

### IMAGENET (32 x 32 min-resized center crops)

Epochs : 200



## Walking in the latent space



### TODO
- [x] model code
- [x] train code
- [ ] add model save code
- [ ] add infer code : load check point
- [ ] add other datasets (LSUN, ImageNet)
- [x] vis results
- [x] add argparse for hyperparams
- [ ] update wandb to log losses 
- [ ] update wandb to log samples