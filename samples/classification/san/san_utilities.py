import numpy
import torch
from torch.nn import init
from PIL import Image
from torch import log_softmax, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd


def step_learning_rate(
    optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6
    ):
  """step learning rate policy"""
  lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9):
  """poly learning rate policy"""
  lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr


def intersection_and_union(output, target, K, ignore_index=255):
  """

:param output:
:type output:
:param target:
:type target:
:param K:
:type K:
:param ignore_index:
:type ignore_index:
:return:
:rtype:
"""
  # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
  assert output.ndim in [1, 2, 3]
  assert output.shape == target.shape
  output = output.reshape(output.size).copy()
  target = target.reshape(target.size)
  output[numpy.where(target == ignore_index)[0]] = 255
  intersection = output[numpy.where(output == target)[0]]
  area_intersection, _ = numpy.histogram(intersection, bins=numpy.arange(K + 1))
  area_output, _ = numpy.histogram(output, bins=numpy.arange(K + 1))
  area_target, _ = numpy.histogram(target, bins=numpy.arange(K + 1))
  area_union = area_output + area_target - area_intersection
  return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, K, ignore_index=255):
  """

:param output:
:type output:
:param target:
:type target:
:param K:
:type K:
:param ignore_index:
:type ignore_index:
:return:
:rtype:
"""
  # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
  assert output.dim() in [1, 2, 3]
  assert output.shape == target.shape
  output = output.view(-1)
  target = target.view(-1)
  output[target == ignore_index] = ignore_index
  intersection = output[output == target]
  # https://github.com/pytorch/pytorch/issues/1382
  area_intersection = torch.histc(
      intersection.float().cpu(), bins=K, min=0, max=K - 1
      )
  area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
  area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
  area_union = area_output + area_target - area_intersection
  return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def init_weights(
    model, conv="kaiming", batchnorm="normal", linear="kaiming", lstm="kaiming"
    ):
  """
:param model: Pytorch Model which is nn.Module
:param conv:  'kaiming' or 'xavier'
:param batchnorm: 'normal' or 'constant'
:param linear: 'kaiming' or 'xavier'
:param lstm: 'kaiming' or 'xavier'
"""
  for m in model.modules():
    if isinstance(m, (_ConvNd)):
      if conv == "kaiming":
        init.kaiming_normal_(m.weight)
      elif conv == "xavier":
        init.xavier_normal_(m.weight)
      else:
        raise ValueError("init type of conv error.\n")
      if m.bias is not None:
        init.constant_(m.bias, 0)

    elif isinstance(m, _BatchNorm):
      if batchnorm == "normal":
        init.normal_(m.weight, 1.0, 0.02)
      elif batchnorm == "constant":
        init.constant_(m.weight, 1.0)
      else:
        raise ValueError("init type of batchnorm error.\n")
      init.constant_(m.bias, 0.0)

    elif isinstance(m, nn.Linear):
      if linear == "kaiming":
        init.kaiming_normal_(m.weight)
      elif linear == "xavier":
        init.xavier_normal_(m.weight)
      else:
        raise ValueError("init type of linear error.\n")
      if m.bias is not None:
        init.constant_(m.bias, 0)

    elif isinstance(m, nn.LSTM):
      for name, param in m.named_parameters():
        if "weight" in name:
          if lstm == "kaiming":
            init.kaiming_normal_(param)
          elif lstm == "xavier":
            init.xavier_normal_(param)
          else:
            raise ValueError("init type of lstm error.\n")
        elif "bias" in name:
          init.constant_(param, 0)


def colorize(gray, palette):
  """

:param gray:
:type gray:
:param palette:
:type palette:
:return:
:rtype:
"""
  # gray: numpy array of the label and 1*3N size list palette
  color = Image.fromarray(gray.astype(numpy.uint8)).convert("P")
  color.putpalette(palette)
  return color


def group_weight(weight_group, module, norm_layer, lr):
  """

:param weight_group:
:type weight_group:
:param module:
:type module:
:param norm_layer:
:type norm_layer:
:param lr:
:type lr:
:return:
:rtype:
"""
  group_decay = []
  group_no_decay = []
  for m in module.modules():
    if isinstance(m, nn.Linear):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)
    elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
      if m.weight is not None:
        group_no_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)

  assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
  weight_group.append(dict(params=group_decay, lr=lr))
  weight_group.append(dict(params=group_no_decay, weight_decay=0.0, lr=lr))
  return weight_group


def group_weight2(weight_group, module, norm_layer, lr):
  """

:param weight_group:
:type weight_group:
:param module:
:type module:
:param norm_layer:
:type norm_layer:
:param lr:
:type lr:
:return:
:rtype:
"""
  group_decay = []
  group_no_decay = []
  for m in module.modules():
    if isinstance(m, nn.Linear):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_decay.append(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_decay.append(m.bias)
    elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
      if m.weight is not None:
        group_no_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)

  assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
  weight_group.append(dict(params=group_decay, lr=lr))
  weight_group.append(dict(params=group_no_decay, weight_decay=0.0, lr=lr))
  return weight_group


def mixup_data(x, y, alpha=0.2):
  """Returns mixed inputs, pairs of targets, and lambda"""
  if alpha > 0:
    lam = numpy.random.beta(alpha, alpha)
  else:
    lam = 1
  index = torch.randperm(x.shape[0])
  x = lam * x + (1 - lam) * x[index, :]
  y_a, y_b = y, y[index]
  return x, y_a, y_b, lam


def mixup_loss(output, target_a, target_b, lam=1.0, eps=0.0):
  """

:param output:
:type output:
:param target_a:
:type target_a:
:param target_b:
:type target_b:
:param lam:
:type lam:
:param eps:
:type eps:
:return:
:rtype:
"""
  w = torch.zeros_like(output).scatter(1, target_a.unsqueeze(1), 1)
  w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
  log_prob = log_softmax(output, dim=1)
  loss_a = (-w * log_prob).sum(dim=1).mean()

  w = torch.zeros_like(output).scatter(1, target_b.unsqueeze(1), 1)
  w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
  log_prob = log_softmax(output, dim=1)
  loss_b = (-w * log_prob).sum(dim=1).mean()
  return lam * loss_a + (1 - lam) * loss_b


def smooth_loss(output, target, eps=0.1):
  """

:param output:
:type output:
:param target:
:type target:
:param eps:
:type eps:
:return:
:rtype:
"""
  w = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
  w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
  log_prob = log_softmax(output, dim=1)
  loss = (-w * log_prob).sum(dim=1).mean()
  return loss


def cal_accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res
