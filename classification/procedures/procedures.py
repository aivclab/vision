import copy
import time

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from classification.processing.data import a_retransform
from segmentation.segmentation_utilities import plot_utilities




def test_model(model, data_iterator, device='cpu'):
  model.eval()

  inputs, labels = next(data_iterator)

  inputs = inputs.to(device)
  labels = labels.to(device)
  with torch.no_grad():
    pred = model(inputs)

  _, predicted = torch.max(pred, 1)[:6]
  pred = pred.data.cpu().numpy()[:6]
  l = labels.cpu().numpy()[:6]

  input_images_rgb = [a_retransform(x) for x in inputs.to('cpu')][:6]

  plot_utilities.plot_prediction(input_images_rgb, l, predicted, pred)
  plt.show()




def train_model(model,
                data_iterator,
                criterion,
                optimizer,
                scheduler,
                writer,
                interrupted_path,
                num_updates=250000,
                early_stop=None):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 1e10
  since = time.time()

  try:
    sess = tqdm(range(num_updates), leave=False)
    for update_i in sess:
      for phase in ['train', 'val']:
        if phase == 'train':
          if scheduler:
            scheduler.step()
            for param_group in optimizer.param_groups:
              writer.add_scalar('lr', param_group['lr'], update_i)

          model.train()
        else:
          model.eval()

        rgb_imgs, true_label = next(data_iterator)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          pred = model(rgb_imgs)
          loss = criterion(pred, true_label)

          if phase == 'train':
            loss.backward()
            optimizer.step()

        update_loss = loss.data.cpu().numpy()
        writer.add_scalar(f'loss/accum', update_loss, update_i)

        if phase == 'val' and update_loss < best_loss:
          best_loss = update_loss
          best_model_wts = copy.deepcopy(model.state_dict())
          writer.add_images(f'rgb_imgs', rgb_imgs, update_i)
          sess.write(f'New best model at update {update_i} with loss {best_loss}')
          torch.save(model.state_dict(), interrupted_path)

      sess.set_description_str(f'Update {update_i} - {phase} accum_loss:{update_loss:2f}')

      if early_stop is not None and update_loss < early_stop:
        break
  except KeyboardInterrupt:
    print('Interrupt')
  finally:
    pass

  model.load_state_dict(best_model_wts)  # load best model weights
  torch.save(model.state_dict(), interrupted_path)

  time_elapsed = time.time() - since
  print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_loss:3f}')

  return model
