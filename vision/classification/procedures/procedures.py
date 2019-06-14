import copy
import string
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from draugr import plot_confusion_matrix
from munin.generate_report import ReportEntry, generate_html, generate_pdf
from munin.utilities.html_embeddings import generate_math_html, plt_html
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from warg.named_ordered_dictionary import NOD

from vision.classification.processing import a_retransform


def test_model(model,
               data_iterator,
               latest_model_path,
               num_columns=2,
               device='cpu'):
  model = model.eval().to(device)

  inputs, labels = next(data_iterator)

  inputs = inputs.to(device)
  labels = labels.to(device)
  with torch.no_grad():
    pred = model(inputs)

  y_pred = pred.data.to(device).numpy()
  y_pred_max = np.argmax(y_pred, axis=-1)
  accuracy_w = accuracy_score(labels, y_pred_max)
  precision_a, recall_a, fscore_a, support_a = precision_recall_fscore_support(labels, y_pred_max)
  precision_w, recall_w, fscore_w, support_w = precision_recall_fscore_support(labels, y_pred_max,
                                                                               average='weighted')

  _, predicted = torch.max(pred, 1)

  truth_labels = labels.data.to(device).numpy()

  input_images_rgb = [a_retransform(x) for x in inputs.to(device)]

  cell_width = (800 / num_columns) - 6 - 6 * 2

  plt.plot(np.random.random((3, 3)))

  alphabet = string.ascii_lowercase
  class_names = np.array([*alphabet])

  samples = len(y_pred)
  predictions = [[None for _ in range(num_columns)] for _ in range(samples // num_columns)]
  for i, a, b, c in zip(range(samples),
                        input_images_rgb,
                        y_pred_max,
                        truth_labels):
    plt.imshow(a)
    if b == c:
      outcome = 'tp'
    else:
      outcome = 'fn'

    gd = ReportEntry(name=i,
                     figure=plt_html(format='jpg',
                                     size=[cell_width, cell_width]),
                     prediction=class_names[b],
                     truth=class_names[c],
                     outcome=outcome)

    predictions[i // num_columns][i % num_columns] = gd

  plot_confusion_matrix(y_pred_max, truth_labels, class_names)

  title = 'Classification Report'
  model_name = latest_model_path
  confusion_matrix = plt_html(format='png', size=[800, 800])

  accuracy = generate_math_html('\dfrac{tp+tn}{N}'), None, accuracy_w
  precision = generate_math_html('\dfrac{tp}{tp+fp}'), precision_a, precision_w
  recall = generate_math_html('\dfrac{tp}{tp+fn}'), recall_a, recall_w
  f1_score = generate_math_html('2*\dfrac{precision*recall}{precision+recall}'), fscore_a, fscore_w
  support = generate_math_html('N_{class_truth}'), support_a, support_w
  metrics = NOD.nod_of(accuracy, precision, f1_score, recall, support).as_flat_tuples()

  bundle = NOD.nod_of(title, model_name, confusion_matrix, metrics, predictions)

  file_name = title.lower().replace(" ", "_")

  generate_html(file_name, **bundle)
  generate_pdf(file_name)

  # plot_utilities.plot_prediction(input_images_rgb, truth_labels, predicted, pred)
  # plt.show()


def confusion_matrisx():
  nb_classes = 9

  cf = torch.zeros(nb_classes, nb_classes)
  with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['val']):
      inputs = inputs.to(device)
      classes = classes.to(device)
      outputs = model_ft(inputs)
      _, preds = torch.max(outputs, 1)
      for t, p in zip(classes.view(-1), preds.view(-1)):
        cf[t.long(), p.long()] += 1

  print(cf)
  print(cf.diag() / cf.sum(1))


def train_model(model,
                data_iterator,
                test_data_iterator,
                criterion,
                optimizer,
                scheduler,
                writer,
                interrupted_path,
                num_updates=250000,
                early_stop=None,
                device='cpu'):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_val_loss = 1e10
  since = time.time()

  try:
    sess = tqdm(range(num_updates), leave=False, disable=False)
    val_loss = 0
    update_loss = 0
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
            # writer.add_images(f'rgb_imgs', test_rgb_imgs, update_i)
            sess.write(f'New best model at update {update_i} with test_loss {best_val_loss}')
            torch.save(model.state_dict(), interrupted_path)

          if early_stop is not None and val_pred < early_stop:
            break
      sess.set_description_str(
          f'Update {update_i} - {phase} accum_loss:{update_loss:2f} test_loss:{val_loss}')

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
