#!/usr/bin/env python
import csv
from pathlib import Path

import PIL.Image
import numpy as np
import torch
import torchvision.transforms
from torch.utils import data


class VggFaces2(data.Dataset):
  mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt

  @staticmethod
  def get_id_label_map(meta_file):
    import pandas
    N_IDENTITY = 9131  # total number of identities in VGG Face2
    N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe
    identity_list = meta_file
    df = pandas.read_csv(identity_list,
                         sep=',\s+',
                         quoting=csv.QUOTE_ALL,
                         encoding="utf-8")
    df["class"] = -1
    df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
    df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)

    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict

  def __init__(self,
               dataset_path: Path,
               image_list_file_path: Path,
               meta_id_path: Path,
               split='train',
               transform: bool = True,
               horizontal_flip: bool = False,
               upper=None):
    """
    :param dataset_path: dataset directory
    :param image_list_file_path: contains image file names under root
    :param id_label_dict: X[class_id] -> label
    :param split: train or valid
    :param transform:
    :param horizontal_flip:
    :param upper: max number of image used for debug
    """
    assert dataset_path.exists(), f"root: {dataset_path} not found."
    self._dataset_path = dataset_path
    assert image_list_file_path.exists(), f"image_list_file: {image_list_file_path} not found."
    self._image_list_file_path = image_list_file_path
    assert meta_id_path.exists(), f'meta id path {meta_id_path} does not exists'
    self._id_label_dict = self.get_id_label_map(meta_id_path)

    self._split = split
    self._transform = transform
    self._horizontal_flip = horizontal_flip

    self._img_info = []
    with open(str(self._image_list_file_path), 'r') as f:
      for i, img_file in enumerate(f):
        img_file = img_file.strip()  # e.g. n004332/0317_01.jpg
        class_id = img_file.split("/")[0]  # like n004332
        label = self._id_label_dict[class_id]
        self._img_info.append({'cid':class_id,
                              'img': img_file,
                              'lbl': label,
                               })
        if i % 1000 == 0:
          print(f"processing: {i} images for {self._split}")
        if upper and i == upper - 1:  # for debug purpose
          break

  def __len__(self):
    return len(self._img_info)

  def __getitem__(self, index):
    info = self._img_info[index]
    img_file = info['img']
    img = PIL.Image.open(str(self._dataset_path / img_file))
    img = torchvision.transforms.Resize(256)(img)

    if self._split == 'train':
      img = torchvision.transforms.RandomCrop(224)(img)
      img = torchvision.transforms.RandomGrayscale(p=0.2)(img)
    else:
      img = torchvision.transforms.CenterCrop(224)(img)

    if self._horizontal_flip:
      img = torchvision.transforms.functional.hflip(img)

    img = np.array(img, dtype=np.uint8)
    assert len(img.shape) == 3  # assumes color images and no alpha channel

    label = info['lbl']
    class_id = info['cid']
    if self._transform:
      return self.transform(img), label, img_file, class_id
    else:
      return img, label, img_file, class_id

  def transform(self, img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= self.mean_bgr
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img

  def untransform(self, img, lbl):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img += self.mean_bgr
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    return img, lbl


if __name__ == '__main__':
  import tqdm

  batch_size = 32

  dt = VggFaces2(Path('/home/heider/Data/vggface2/test'),
                 Path('/home/heider/Data/vggface2/test_list.txt'),
                 Path('/home/heider/Data/vggface2/identity_meta.csv'),
                 split='test')

  test_loader = torch.utils.data.DataLoader(dt,
                                            batch_size=batch_size,
                                            shuffle=True)

  for batch_idx, (imgs, target, img_files, class_ids) in tqdm.tqdm(enumerate(test_loader),
                                                                   total=len(test_loader),
                                                                   desc='Bro',
                                                                   ncols=80,
                                                                   leave=False):
    print(imgs)
    break
