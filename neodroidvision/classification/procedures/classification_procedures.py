import copy
import string
import time
from pathlib import Path

import numpy
import torch
import tqdm
from draugr.numpy_utilities import SplitEnum
from draugr.python_utilities import (
    rgb_drop_alpha_batch_nhwc,
    torch_vision_normalize_batch_nchw,
)
from draugr.torch_utilities import (
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
    to_tensor,
    uint_nhwc_to_nchw_float_batch,
)
from draugr.visualisation import confusion_matrix_plot
from matplotlib import pyplot
from munin.generate_report import ReportEntry, generate_html, generate_pdf
from munin.html_embeddings import generate_math_html, plt_html
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from warg import NOD

__all__ = ["test_model", "pred_target_train_model"]

from neodroidvision.data.neodroid_environments.classification.data import (
    default_torch_retransform,
)


def test_model(model, data_iterator, latest_model_path, num_columns: int = 2):
    model = model.eval().to(global_torch_device())

    inputs, labels = next(data_iterator)

    inputs = inputs.to(global_torch_device())
    labels = labels.to(global_torch_device())
    with torch.no_grad():
        pred = model(inputs)

    y_pred = pred.data.to("cpu").numpy()
    y_pred_max = numpy.argmax(y_pred, axis=-1)
    accuracy_w = accuracy_score(labels, y_pred_max)
    precision_a, recall_a, fscore_a, support_a = precision_recall_fscore_support(
        labels, y_pred_max
    )
    precision_w, recall_w, fscore_w, support_w = precision_recall_fscore_support(
        labels, y_pred_max, average="weighted"
    )

    _, predicted = torch.max(pred, 1)

    truth_labels = labels.data.to("cpu").numpy()

    input_images_rgb = [
        default_torch_retransform(x) for x in inputs.to(global_torch_device())
    ]

    cell_width = (800 / num_columns) - 6 - 6 * 2

    pyplot.plot(numpy.random.random((3, 3)))

    alphabet = string.ascii_lowercase
    class_names = numpy.array([*alphabet])

    samples = len(y_pred)
    predictions = [
        [None for _ in range(num_columns)] for _ in range(samples // num_columns)
    ]
    for i, a, b, c in zip(range(samples), input_images_rgb, y_pred_max, truth_labels):
        pyplot.imshow(a)
        if b == c:
            outcome = "tp"
        else:
            outcome = "fn"

        gd = ReportEntry(
            name=i,
            figure=plt_html(a, format="jpg", size=(cell_width, cell_width)),
            prediction=class_names[b],
            truth=class_names[c],
            outcome=outcome,
            explanation=None,
        )

        predictions[i // num_columns][i % num_columns] = gd

    cfmat = confusion_matrix_plot(y_pred_max, truth_labels, class_names)

    title = "Classification Report"
    model_name = latest_model_path
    confusion_matrix = plt_html(cfmat, format="png", size=(800, 800))

    accuracy = generate_math_html("\dfrac{tp+tn}{N}"), None, accuracy_w
    precision = generate_math_html("\dfrac{tp}{tp+fp}"), precision_a, precision_w
    recall = generate_math_html("\dfrac{tp}{tp+fn}"), recall_a, recall_w
    f1_score = (
        generate_math_html("2*\dfrac{precision*recall}{precision+recall}"),
        fscore_a,
        fscore_w,
    )
    support = generate_math_html("N_{class_truth}"), support_a, support_w
    metrics = NOD.nod_of(
        accuracy, precision, f1_score, recall, support
    ).as_flat_tuples()

    bundle = NOD.nod_of(title, model_name, confusion_matrix, metrics, predictions)

    file_name = Path(title.lower().replace(" ", "_"))

    generate_html(file_name.with_suffix(".html"), **bundle)
    generate_pdf(file_name.with_suffix(".html"), file_name.with_suffix(".pdf"))

    # plot_utilities.plot_prediction(input_images_rgb, truth_labels, predicted, pred)
    # pyplot.show()


def pred_target_train_model(
    model,
    train_iterator,
    criterion,
    optimiser,
    scheduler,
    writer,
    interrupted_path,
    test_data_iterator=None,
    num_updates: int = 250000,
    early_stop=None,
) -> torch.nn.Module:
    """

    Args:
      model:
      train_iterator:
      criterion:
      optimiser:
      scheduler:
      writer:
      interrupted_path:
      test_data_iterator:
      num_updates:
      early_stop:

    Returns:

    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 1e10
    since = time.time()

    try:
        sess = tqdm.tqdm(range(num_updates), leave=False, disable=False)
        val_loss = 0
        update_loss = 0
        val_acc = 0
        last_val = None
        last_out = None
        with torch.autograd.detect_anomaly():
            for update_i in sess:
                for phase in [SplitEnum.training, SplitEnum.validation]:
                    if phase == SplitEnum.training:
                        with TorchTrainSession(model):

                            input, true_label = zip(*next(train_iterator))

                            rgb_imgs = torch_vision_normalize_batch_nchw(
                                uint_nhwc_to_nchw_float_batch(
                                    rgb_drop_alpha_batch_nhwc(to_tensor(input))
                                )
                            )
                            true_label = to_tensor(true_label, dtype=torch.long)
                            optimiser.zero_grad()

                            pred = model(rgb_imgs)
                            loss = criterion(pred, true_label)
                            loss.backward()
                            optimiser.step()

                            if last_out is None:
                                last_out = pred
                            else:
                                if not torch.dist(last_out, pred) > 0:
                                    print(f"Same output{last_out},{pred}")
                                last_out = pred

                            update_loss = loss.data.cpu().numpy()
                            writer.scalar(f"loss/train", update_loss, update_i)

                            if scheduler:
                                scheduler.step()
                    elif test_data_iterator:
                        with TorchEvalSession(model):
                            test_rgb_imgs, test_true_label = zip(*next(train_iterator))
                            test_rgb_imgs = torch_vision_normalize_batch_nchw(
                                uint_nhwc_to_nchw_float_batch(
                                    rgb_drop_alpha_batch_nhwc(to_tensor(test_rgb_imgs))
                                )
                            )

                            test_true_label = to_tensor(
                                test_true_label, dtype=torch.long
                            )

                            with torch.no_grad():
                                val_pred = model(test_rgb_imgs)
                                val_loss = criterion(val_pred, test_true_label)

                            _, cat = torch.max(val_pred, -1)
                            val_acc = torch.sum(cat == test_true_label) / float(
                                cat.size(0)
                            )
                            writer.scalar(f"loss/acc", val_acc, update_i)
                            writer.scalar(f"loss/val", val_loss, update_i)

                            if last_val is None:
                                last_val = cat
                            else:
                                if all(last_val == cat):
                                    print(f"Same val{last_val},{cat}")
                                last_val = cat

                            if val_loss < best_val_loss:
                                best_val_loss = val_loss

                                best_model_wts = copy.deepcopy(model.state_dict())
                                sess.write(
                                    f"New best validation model at update {update_i} with test_loss {best_val_loss}"
                                )
                                torch.save(model.state_dict(), interrupted_path)

                        if early_stop is not None and val_pred < early_stop:
                            break
                sess.set_description_str(
                    f"Update {update_i} - {phase} "
                    f"update_loss:{update_loss:2f} "
                    f"test_loss:{val_loss}"
                    f"val_acc:{val_acc}"
                )

    except KeyboardInterrupt:
        print("Interrupt")
    finally:
        pass

    model.load_state_dict(best_model_wts)  # load best model weights

    time_elapsed = time.time() - since
    print(f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_val_loss:3f}")

    return model
