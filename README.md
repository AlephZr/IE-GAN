# IE-GAN[[arXiv]](https://arxiv.org/abs/2109.11078)
Codes for paper "IE-GAN: An Improved Evolutionary Generative Adversarial Network Using a New Fitness Function and a Generic Crossover Operator"

## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Preparation
- Preparing *.npz* files for Pytorch Inception metrics evaluation (cifar10 as an example):
```
python metric/inception_pytorch/calculate_inception_moments.py --dataset C10 --data_root datasets
```
- Preparing *.tgz* files for Tensorflow Inception metrics evaluation
- Preparing *.npz* files for Tensorflow FID metrics evaluation

### Operator GANs Training

An example of [GAN](https://arxiv.org/abs/1406.2661) training command was saved in [./scripts/operator_test.sh](). Train a model (cifar10 as an example): 
```bash
bash ./scripts/operator_test.sh
```

### IE-GAN Training

An example of IE-GAN training command was saved in [./scripts/CIFAR10_32.sh](). Train a model (cifar10 as an example):
```bash
bash ./scripts/CIFAR10_32.sh
```

## Acknowledgments

- The authors would like to thank  Dr. [Chaoyue Wang](https://www.sydney.edu.au/engineering/about/our-people/academic-staff/chaoyue-wang.html) for his assistance.

- Pytorch Inception metrics code from [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).

- TensorFlow Inception Score code from [OpenAI's Improved-GAN.](https://github.com/openai/improved-gan).

- TensorFlow FID code from [TTUR](https://github.com/bioinf-jku/TTUR).

