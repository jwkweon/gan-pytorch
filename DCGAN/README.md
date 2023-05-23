# PyTorch implementation of [DCGAN](https://arxiv.org/pdf/1511.06434.pdf).
UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS.

## Modifications for Implementation
To ensure output size (64 x 64), I set kernel size, stride, padding size to 4, 2, 1, respectively.


## Run Code

```ShellSession
$ python train.py
```

## DDP MULTI-GPUs Training

if you have multi-gpus using it to train more faster!
I use it for imagenet.

```ShellSession
$ python -m torch.distributed.launch --nproc_per_node={num gpus} --master_port={port} train_ddp.py --dataset_name 'imagenet' --dataset_path {your dataset path} --n_batch {batchsize}
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
- [x] add model save code
- [x] add load check point
- [ ] add infer code : generate samples
- [x] add other datasets (ImageNet)
- [ ] add other datasets (LSUN)
- [x] vis results
- [x] add argparse for hyperparams
- [ ] update wandb to log losses 
- [ ] update wandb to log samples
- [x] tqdm : progress bar
- [x] ddp : support multi-gpu training