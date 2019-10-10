import os
import time
from collections import defaultdict

from matplotlib import pyplot
import pandas as pd
import seaborn as sns
import torch
from objectives import loss_fn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.reconstruction import ConditionalVAE
from warg.named_ordered_dictionary import NOD

fig_root = PROJECT_APP_PATH.user_data / 'cvae'

args = NOD()
args.seed = 58329583
args.epochs = 1000
args.batch_size = 256
args.learning_rate = 0.001
args.encoder_layer_sizes = [784, 256]
args.decoder_layer_sizes = [256, 784]
args.latent_size = 10
args.print_every = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timstamp = time.time()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(args.seed)

vae = ConditionalVAE(encoder_layer_sizes=args.encoder_layer_sizes,
                     latent_size=args.latent_size,
                     decoder_layer_sizes=args.decoder_layer_sizes,
                     num_conditions=10).to(DEVICE)
dataset = MNIST(root=str(PROJECT_APP_PATH.user_data / 'MNIST'),
                train=True,
                transform=transforms.ToTensor(),
                download=True)


def one_hot(labels, num_labels, device='cpu'):
  targets = torch.zeros(labels.size(0), num_labels)
  for i, label in enumerate(labels):
    targets[i, label] = 1
  return targets.to(device=device)


def main():
  data_loader = DataLoader(dataset=dataset,
                           batch_size=args.batch_size,
                           shuffle=True)

  optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

  logs = defaultdict(list)

  for epoch in range(args.epochs):
    tracker_epoch = defaultdict(lambda:defaultdict(dict))

    for iteration, (original, label) in enumerate(data_loader):

      original, label = original.to(DEVICE), label.to(DEVICE)
      reconstruction, mean, log_var, z = vae(original, one_hot(label,
                                                               10,
                                                               device=DEVICE))

      for i, yi in enumerate(label):
        id = len(tracker_epoch)
        tracker_epoch[id]['x'] = z[i, 0].item()
        tracker_epoch[id]['y'] = z[i, 1].item()
        tracker_epoch[id]['label'] = yi.item()

      optimizer.zero_grad()
      loss = loss_fn(reconstruction, original, mean, log_var)
      loss.backward()
      optimizer.step()

      logs['loss'].append(loss.item())

      if iteration % args.print_every == 0 or iteration == len(data_loader) - 1:
        print(f"Epoch {epoch:02d}/{args.epochs:02d}"
              f" Batch {iteration:04d}/{len(data_loader) - 1:d},"
              f" Loss {loss.item():9.4f}")

        condition_vector = torch.arange(0, 10, device=DEVICE).long().unsqueeze(1)
        sample = vae.sample(one_hot(condition_vector, 10, device=DEVICE), num=condition_vector.size(0))

        pyplot.figure()
        pyplot.figure(figsize=(5, 10))
        for p in range(10):
          pyplot.subplot(5, 2, p + 1)

          pyplot.text(0, 0, f"c={condition_vector[p].item():d}",
                   color='black',
                   backgroundcolor='white',
                   fontsize=8)
          pyplot.imshow(sample[p].cpu().data.numpy())
          pyplot.axis('off')

        if not os.path.exists(os.path.join(fig_root, str(timstamp))):
          if not (os.path.exists(os.path.join(fig_root))):
            os.mkdir(os.path.join(fig_root))
          os.mkdir(os.path.join(fig_root, str(timstamp)))

        pyplot.savefig(os.path.join(fig_root, str(timstamp),
                                 f"Epoch{epoch:d}_Iter{iteration:d}.png"),
                    dpi=300)
        pyplot.clf()
        pyplot.close('all')

    df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
    g = sns.lmplot(x='x',
                   y='y',
                   hue='label',
                   data=df.groupby('label').head(100),
                   fit_reg=False,
                   legend=True)
    g.savefig(os.path.join(fig_root,
                           str(timstamp),
                           f"Epoch{epoch:d}_latent_space.png"),
              dpi=300)


if __name__ == '__main__':

  main()
