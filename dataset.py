import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import random_split


def get_MNIST_data(train_split: float):
	traindata = MNIST("data", train=True, download=True, transform=ToTensor())
	testdata = MNIST("data", train=False, download=True, transform=ToTensor())
	(traindata, valdata) = random_split(
		traindata,
		[int(len(traindata) * train_split), int(len(traindata) * (1 - train_split))],
		generator=torch.Generator().manual_seed(42)
	)
	return traindata, testdata, valdata