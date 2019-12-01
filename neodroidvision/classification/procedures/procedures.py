import copy
import string
import time

from matplotlib import pyplot
import numpy
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from draugr import global_torch_device, to_tensor, uint_hwc_to_chw_float_batch, torch_vision_normalize_chw
from draugr.torch_utilities.images.channel_transform import rgb_drop_alpha_batch
from draugr.visualisation import plot_confusion_matrix
from munin.generate_report import ReportEntry, generate_html, generate_pdf
from munin.utilities.html_embeddings import generate_math_html, plt_html
from neodroidvision.classification.processing import a_retransform
from warg.named_ordered_dictionary import NOD


def test_model(model,
               data_iterator,
               latest_model_path,
               num_columns=2):
  model = model.eval().to(global_torch_device())

  inputs, labels = next(data_iterator)

  inputs = inputs.to(global_torch_device())
  labels = labels.to(global_torch_device())
  with torch.no_grad():
    pred = model(inputs)

  y_pred = pred.data.to('cpu').numpy()
  y_pred_max = numpy.argmax(y_pred, axis=-1)
  accuracy_w = accuracy_score(labels, y_pred_max)
  precision_a, recall_a, fscore_a, support_a = precision_recall_fscore_support(labels, y_pred_max)
  precision_w, recall_w, fscore_w, support_w = precision_recall_fscore_support(labels, y_pred_max,
                                                                               average='weighted')

  _, predicted = torch.max(pred, 1)

  truth_labels = labels.data.to('cpu').numpy()

  input_images_rgb = [a_retransform(x) for x in inputs.to(global_torch_device())]

  cell_width = (800 / num_columns) - 6 - 6 * 2

  pyplot.plot(numpy.random.random((3, 3)))

  alphabet = string.ascii_lowercase
  class_names = numpy.array([*alphabet])

  samples = len(y_pred)
  predictions = [[None for _ in range(num_columns)] for _ in range(samples // num_columns)]
  for i, a, b, c in zip(range(samples),
                        input_images_rgb,
                        y_pred_max,
                        truth_labels):
    pyplot.imshow(a)
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
  # pyplot.show()


def pred_target_train_model(model,
                            train_iterator,
                            criterion,
                            optimizer,
                            scheduler,
                            writer,
                            interrupted_path,
                            test_data_iterator=None,
                            num_updates=250000,
                            early_stop=None):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_val_loss = 1e10
  since = time.time()

  try:
    sess = tqdm(range(num_updates), leave=False, disable=False)
    val_loss = 0
    update_loss = 0
    val_acc = 0
    last_val = None
    last_out = None
    with torch.autograd.detect_anomaly():
      for update_i in sess:
        for phase in ['train', 'val']:
          if phase == 'train':
            # model.train()

            input, true_label = zip(*next(train_iterator))

            rgb_imgs = torch_vision_normalize_chw(uint_hwc_to_chw_float_batch(
              rgb_drop_alpha_batch(to_tensor(input))
              ))
            true_label = to_tensor(true_label, dtype=torch.long)
            optimizer.zero_grad()

            pred = model(rgb_imgs)
            loss = criterion(pred, true_label)
            loss.backward()
            optimizer.step()

            if last_out is None:
              last_out = pred
            else:
              if not torch.dist(last_out, pred) > 0:
                print(f'Same output{last_out},{pred}')
              last_out = pred

            update_loss = loss.data.cpu().numpy()
            writer.scalar(f'loss/train', update_loss, update_i)

            if scheduler:
              scheduler.step()
          elif test_data_iterator:
            # model.eval()

            test_rgb_imgs, test_true_label = zip(*next(train_iterator))
            test_rgb_imgs = torch_vision_normalize_chw(uint_hwc_to_chw_float_batch(rgb_drop_alpha_batch(to_tensor(
              test_rgb_imgs))))

            test_true_label = to_tensor(test_true_label, dtype=torch.long)

            with torch.no_grad():
              val_pred = model(test_rgb_imgs)
              val_loss = criterion(val_pred, test_true_label)

            _, cat = torch.max(val_pred, -1)
            val_acc = torch.sum(cat == test_true_label) / float(cat.size(0))
            writer.scalar(f'loss/acc', val_acc, update_i)
            writer.scalar(f'loss/val', val_loss, update_i)

            if last_val is None:
              last_val = cat
            else:
              if (all(last_val == cat)):
                print(f'Same val{last_val},{cat}')
              last_val = cat

            if val_loss < best_val_loss:
              best_val_loss = val_loss

              best_model_wts = copy.deepcopy(model.state_dict())
              sess.write(f'New best validation model at update {update_i} with test_loss {best_val_loss}')
              torch.save(model.state_dict(), interrupted_path)

            if early_stop is not None and val_pred < early_stop:
              break
        sess.set_description_str(
          f'Update {update_i} - {phase} '
          f'update_loss:{update_loss:2f} '
          f'test_loss:{val_loss}'
          f'val_acc:{val_acc}')

  except KeyboardInterrupt:
    print('Interrupt')
  finally:
    pass

  model.load_state_dict(best_model_wts)  # load best model weights

  time_elapsed = time.time() - since
  print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_val_loss:3f}')

  return model
