# %%
import numpy
import torch
import torch.utils.data

# %%
# %matplotlib inline
from matplotlib import pyplot
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional
from torchvision import datasets, transforms
from torchvision.utils import make_grid


def show_adn_save(file_name, img):
    npimg = numpy.transpose(img.numpy(), (1, 2, 0))
    f = f"./{file_name}.png"
    pyplot.imshow(npimg)
    pyplot.imsave(f, npimg)


# %%
class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hin=500, k=5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_hin, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k

    def sample_from_p(self, p):
        return functional.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def v_to_h(self, v):
        p_h = functional.sigmoid(functional.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = functional.sigmoid(functional.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)

        return v, v_

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = functional.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.log().sum(1)
        return (-hidden_term - vbias_term).mean()


# %%
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    ),
    batch_size=batch_size,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data", train=False, transform=transforms.Compose([transforms.ToTensor()])
    ),
    batch_size=batch_size,
)

# %%
rbm = RBM(k=1)

# %%
train_op = optim.SGD(rbm.parameters(), 0.1)

# %%
for epoch in range(10):
    loss_ = []
    for _, (data, target) in enumerate(train_loader):
        data = Variable(data.view(-1, 784))
        sample_data = data.bernoulli()

        v, v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data[0])
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    print(numpy.mean(loss_))

# %%
show_adn_save("real", make_grid(v.view(32, 1, 28, 28).data))

# %%
show_adn_save("generate", make_grid(v1.view(32, 1, 28, 28).data))

# %%
