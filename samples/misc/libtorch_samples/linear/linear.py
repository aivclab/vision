import torch
from matplotlib import pyplot
from torch.utils.data import DataLoader, TensorDataset

input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

x_train = torch.Tensor(
    [[2.1], [3.3], [3.6], [4.4], [5.5], [6.3], [6.5], [7.0], [7.5], [9.7]]
)
y_train = torch.Tensor(
    [[1.0], [1.2], [1.9], [2.0], [2.5], [2.5], [2.2], [2.7], [3.0], [3.6]]
)
dataset = TensorDataset(x_train, y_train)
data_loader = DataLoader(dataset)

model = torch.nn.Linear(input_size, output_size)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for example, label in data_loader:
        prediction = model.forward(example)
        loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

predicted = model(x_train).detach()
pyplot.plot(x_train.numpy(), y_train.numpy(), "ro", label="Original data")
pyplot.plot(x_train.numpy(), predicted.numpy(), label="Fitted line")
pyplot.legend()
pyplot.show()
