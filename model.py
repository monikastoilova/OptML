import torch.nn as nn
import torch


class CNN(nn.Module):
	def __init__(self, layers: list[dict], lin_features, num_classes):
		super(CNN, self).__init__()
		self.conv_layers = []
		self.relu_layers = []
		self.maxp_layers = []
		for l in layers:
			self.conv_layers.append(
				nn.Conv2d(in_channels=l["in_ch"], out_channels=l["out_ch"], kernel_size=(l["conv_k_sz"], l["conv_k_sz"]), padding="same")
			)
			self.relu_layers.append(
				nn.ReLU()
			)
			self.maxp_layers.append(
				nn.MaxPool2d(kernel_size=(l["maxp_k_sz"], l["maxp_k_sz"]), stride=(l["maxp_str"], l["maxp_str"]))
			)
		# initialize first (and only) set of FC => RELU layers
		self.lin1 = nn.Linear(in_features=lin_features, out_features=500)
		self.relu1 = nn.ReLU()
		# initialize our softmax classifier
		self.lin2 = nn.Linear(in_features=500, out_features=num_classes)
		self.logSoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		for conv, relu, maxp in zip(self.conv_layers, self.relu_layers, self.maxp_layers):
			x = conv(x)
			x = relu(x)
			x = maxp(x)
		x = torch.flatten(x, 1)
		x = self.lin1(x)
		x = self.relu1(x)
		x = self.lin2(x)
		x = self.logSoftmax(x)
		return x