import numpy as np
import torch
from torch import nn
from collections import OrderedDict


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(OrderedDict([('fc1', nn.Linear(5, 15)),
                                                ('fc2', nn.Sigmoid()),
                                                ('fc3', nn.Linear(15, 5)),
                                                ('fc4', nn.Sigmoid()),
                                                ('fc5', nn.Linear(5, 4)),
                                                ('fc6', nn.LogSoftmax())]))

    def forward(self, x):
        return self.model(x)
"""
    Features :
    1. the number of tracepoints that were traced
    2. the cumulative moving average of page offsets
    3. the cumulative moving standard deviation of page offsets
    4. the mean absolute page offset differences for consecutive tracepoints
    5. the current readahead value
"""

# generate random training set of length 100

training_x = np.random.random((100, 5))

# Possible output classifications : 0, 1, 2, 3

y = np.random.randint(0, 4, size=100)

# normalize data using z-score
means = np.mean(training_x, axis=0)
std_devs = np.std(training_x, axis=0)
training_x = (training_x - means) / std_devs

# model
model = NeuralNetwork()
loss_fn = nn.NLLLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
for e in range(10):
    output = model.forward(torch.tensor(training_x, dtype=torch.float))
    loss = loss_fn(output, torch.tensor(y, dtype=torch.long))
    loss.backward()
    opt.step()
    print("loss = ", loss)




