import copy
import time

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from classification.processing.data import a_retransform
from segmentation.segmentation_utilities import plot_utilities


def test_model(model, data_iterator,criterion=None, device='cpu'):
  model.eval()

  inputs, labels = next(data_iterator)

  inputs = inputs.to(device)
  labels = labels.to(device)
  with torch.no_grad():
    pred = model(inputs)


  if criterion:
    loss = criterion(pred, labels)
    print(loss)


  y_pred = pred.data.numpy()
  accuracy = accuracy_score(labels, np.argmax(y_pred, axis=1))
  print(accuracy)

  _, predicted = torch.max(pred, 1)[:6]
  pred = pred.data.cpu().numpy()[:6]
  l = labels.cpu().numpy()[:6]

  input_images_rgb = [a_retransform(x) for x in inputs.to('cpu')][:6]

  plot_utilities.plot_prediction(input_images_rgb, l, predicted, pred)
  plt.show()


def confusion_matrix():
  nb_classes = 9

  confusion_matrix = torch.zeros(nb_classes, nb_classes)
  with torch.no_grad():
      for i, (inputs, classes) in enumerate(dataloaders['val']):
          inputs = inputs.to(device)
          classes = classes.to(device)
          outputs = model_ft(inputs)
          _, preds = torch.max(outputs, 1)
          for t, p in zip(classes.view(-1), preds.view(-1)):
                  confusion_matrix[t.long(), p.long()] += 1

  print(confusion_matrix)
  print(confusion_matrix.diag() / confusion_matrix.sum(1))

def train_model(model,
                data_iterator,
                test_data_iterator,
                criterion,
                optimizer,
                scheduler,
                writer,
                interrupted_path,
                num_updates=250000,
                early_stop=None,device='cpu'):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_val_loss = 1e10
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

          rgb_imgs, true_label = next(data_iterator)

          optimizer.zero_grad()

          with torch.set_grad_enabled(phase == 'train'):
            pred = model(rgb_imgs)
            loss = criterion(pred, true_label)

            if phase == 'train':
              loss.backward()
              optimizer.step()

          update_loss = loss.data.cpu().numpy()
          writer.add_scalar(f'loss/train', update_loss, update_i)

          sess.set_description_str(f'Update {update_i} - {phase} accum_loss:{update_loss:2f}')

        else:
          model.eval()

          test_rgb_imgs, test_true_label = next(test_data_iterator)
          test_rgb_imgs, test_true_label = test_rgb_imgs.to(device), test_true_label.to(device)
          with torch.set_grad_enabled(False):
            val_pred = model(test_rgb_imgs)
            val_loss = criterion(val_pred, test_true_label)

          writer.add_scalar(f'loss/val', val_loss, update_i)
          if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            #writer.add_images(f'rgb_imgs', test_rgb_imgs, update_i)
            sess.write(f'New best model at update {update_i} with test_loss {best_val_loss}')
            torch.save(model.state_dict(), interrupted_path)

          if early_stop is not None and val_pred < early_stop:
            break

  except KeyboardInterrupt:
    print('Interrupt')
  finally:
    pass

  model.load_state_dict(best_model_wts)  # load best model weights
  torch.save(model.state_dict(), interrupted_path)

  time_elapsed = time.time() - since
  print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_val_loss:3f}')

  return model
