#!/usr/bin/env python
import csv
from pathlib import Path

from matplotlib import pyplot
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class VggFaces2(data.Dataset):
  # mean = numpy.array([0.485, 0.456, 0.406])
  # std = numpy.array([0.229, 0.224, 0.225])

  inverse_transform = transforms.Compose([
    # transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
    transforms.ToPILImage()
    ])

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
               split: str = 'train',
               resize_s=256,
               raw_images=False):
    """
    :type resize_s: int or tuple(w,h)
    :param dataset_path: dataset directory
    :param split: train, valid, test
    """
    assert dataset_path.exists(), f"root: {dataset_path} not found."
    self._dataset_path = dataset_path / split
    image_list_file_path = dataset_path / f'{split}_list.txt'
    assert (image_list_file_path.exists(),
            f"image_list_file: {image_list_file_path} not found.")
    self._image_list_file_path = image_list_file_path
    meta_id_path = dataset_path / 'identity_meta.csv'
    assert meta_id_path.exists(), f'meta id path {meta_id_path} does not exists'
    self._split = split
    self._id_label_dict = self.get_id_label_map(meta_id_path)
    self._raw_images = raw_images

    self.train_trans = transforms.Compose([
      transforms.RandomResizedCrop(resize_s),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      # transforms.Normalize(self.mean, self.std)
      ])

    self.val_trans = transforms.Compose([
      transforms.Resize(resize_s),
      transforms.CenterCrop(resize_s),
      transforms.ToTensor(),
      # transforms.Normalize(self.mean, self.std)
      ])

    self._img_info = []
    with open(str(self._image_list_file_path), 'r') as f:
      for i, img_file in enumerate(f):
        img_file = img_file.strip()  # e.g. n004332/0317_01.jpg
        class_id = img_file.split("/")[0]  # like n004332
        label = self._id_label_dict[class_id]
        self._img_info.append({'class_id':class_id,
                               'img':     img_file,
                               'label':   label,
                               })
        if i % 1000 == 0:
          print(f"Processing: {i} images for {self._split} split")

  def __len__(self):
    return len(self._img_info)

  def __getitem__(self, index):
    info = self._img_info[index]
    img_file = info['img']
    img = Image.open(str(self._dataset_path / img_file))

    if not self._raw_images:
      if self._split == 'train':
        img = self.train_trans(img)
      else:
        img = self.val_trans(img)

    label = info['label']
    class_id = info['class_id']

    return img, label, img_file, class_id


if __name__ == '__main__':
  import tqdm

  batch_size = 32

  dt = VggFaces2(Path('/home/heider/Data/vggface2'),
                 split='test',
                 # raw_images=True
                 )

  test_loader = torch.utils.data.DataLoader(dt,
                                            batch_size=batch_size,
                                            shuffle=False)

  # test_loader = dt

  for batch_idx, (imgs, label, img_files, class_ids) in tqdm.tqdm(enumerate(test_loader),
                                                                  total=len(test_loader),
                                                                  desc='Bro',
                                                                  ncols=80,
                                                                  leave=False):
    pyplot.imshow(dt.inverse_transform(imgs[0]))
    # pyplot.imshow(imgs)
    pyplot.show()
    break
