# PyTorch implementation of [GAN](https://arxiv.org/abs/1406.2661).

## Modifications for Implementation
The provided model implementation includes three datasets: MNIST, FashionMNIST, and CIFAR10. For convenience, the output image size of the model has been changed to 32x32 for all datasets, and more information can be found in the gan.py file. 
Additionally, the maxout activation function in the discriminator has been replaced with sigmoid. 
Please note that the layer parameters and their numbers in the model were arbitrarily determined, and you are free to change them as needed for your convenience.


## Run Code

```python
python train.py
```


## Generated Images
### MNIST
<img src=results\mnist\200_gif_results_mnist.gif>

Epochs : 200

### FashionMNIST
<img src=results\fashion\200_gif_results_fashion.gif>

Epochs : 200

### CIFAR10
<img src=results\cifar10\200_gif_results_cifar10.gif>

Epochs : 200

<img src=results\cifar10\500_gif_results_cifar10.gif>

Epochs : 500

### TODO
- [x] model code
- [x] train code
- [ ] add model save code
- [ ] add infer code : load check point
- [x] vis results
- [ ] add argparse for hyperparams
- [ ] update wandb to log losses 
