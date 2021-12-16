import gzip
import pickle
from pathlib import Path

import h5py
import numpy
import numpy as np
from draugr.numpy_utilities import SplitEnum
from tqdm import trange

from .augmentation import rotate_y

MIN_X, MAX_X = (-0.5, 0.5)
MIN_Y, MAX_Y = (-0.5, 0.5)
MIN_Z, MAX_Z = (-3, 3)

N_X = 5
N_Y = 5
N_Z = 30


def make_voxel() -> numpy.ndarray:
  """

  Returns:

  """
  # Define voxel parameters

  # VOXEL CREATION
  # with normals

  front = np.array(np.meshgrid(np.linspace(MIN_X, MAX_X, N_X),
                               np.linspace(MIN_Y, MAX_Y, N_Y),
                               MAX_Z)).T.reshape(-1, 3)
  front = np.concatenate((front, [[1, 0, 0]] * len(front)), axis=1)

  back = np.array(np.meshgrid(np.linspace(MIN_X, MAX_X, N_X),
                              np.linspace(MIN_Y, MAX_Y, N_Y),
                              MIN_Z)).T.reshape(-1, 3)
  back = np.concatenate((back, [[-1, 0, 0]] * len(back)), axis=1)

  top = np.array(np.meshgrid(MIN_X,
                             np.linspace(MIN_Y, MAX_Y, N_Y),
                             np.linspace(MIN_Z, MAX_Z, N_Z))).T.reshape(-1, 3)
  top = np.concatenate((top, [[0, 0, 1]] * len(top)), axis=1)

  bottom = np.array(np.meshgrid(MAX_X,
                                np.linspace(MIN_Y, MAX_Y, N_Y),
                                np.linspace(MIN_Z, MAX_Z, N_Z))).T.reshape(-1, 3)
  bottom = np.concatenate((bottom, [[0, 0, -1]] * len(bottom)), axis=1)

  left = np.array(np.meshgrid(np.linspace(MIN_X, MAX_X, N_X),
                              MIN_Y,
                              np.linspace(MIN_Z, MAX_Z, N_Z))).T.reshape(-1, 3)
  left = np.concatenate((left, [[0, -1, 0]] * len(left)), axis=1)

  right = np.array(np.meshgrid(np.linspace(MIN_X, MAX_X, N_X),
                               MAX_Y,
                               np.linspace(MIN_Z, MAX_Z, N_Z))).T.reshape(-1, 3)
  right = np.concatenate((right, [[0, 1, 0]] * len(right)), axis=1)

  voxel = np.array((front, back, top, bottom, left, right), dtype=object)
  return voxel


def img_to_point_cloud(input_image, voxel):
  """

  Args:
    input_image:
    voxel:

  Returns:

  """
  non_zero_coord = np.transpose(np.nonzero(input_image))

  # dict for fast looking of neighbor occupancy
  non_zero_dict = {}
  for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
      non_zero_dict[str([i, j])] = any(np.all([i, j] == non_zero_coord, axis=1))

  cloud = []

  for n in range(len(non_zero_coord)):
    x = non_zero_coord[n][0]
    y = non_zero_coord[n][1]

    components = [0, 1]

    # top
    if not non_zero_dict[str([x - 1, y])]:
      components.append(2)

    # bottom
    if not non_zero_dict[str([x + 1, y])]:
      components.append(3)

    # left
    if not non_zero_dict[str([x, y - 1])]:
      components.append(4)

    # right
    if not non_zero_dict[str([x, y + 1])]:
      components.append(5)

    pixel_cloud = np.concatenate(voxel[components])

    # move the voxel to its position
    pixel_cloud[:, 0] += x
    pixel_cloud[:, 1] += y

    cloud.append(pixel_cloud)

  cloud = np.concatenate(cloud)

  xyzmin = np.min(cloud[:, :3], axis=0)
  xyzmax = np.max(cloud[:, :3], axis=0)
  diff = xyzmax - xyzmin
  cloud[:, :3] = ((cloud[:, :3] - xyzmin[np.argmax(diff)]) / diff[np.argmax(diff)])  # make max range 0-1

  cloud[:, :3] -= np.mean(cloud[:, :3], axis=0)  # 0 mean

  return cloud




def save_dataset(X, y, voxel, output, shape=(28, 28)):
  """

  Args:
    X:
    y:
    voxel:
    output:
    shape:
  """
  img = np.zeros((shape[0] + 2, shape[1] + 2))

  with h5py.File(output.with_suffix('.h5'), "w") as hf:
    for i in trange(len(X)):
      img[1:-1, 1:-1] = X[i].reshape(shape[0], shape[1])
      data = img_to_point_cloud(img, voxel)

      # rotate to vertical
      transf = np.c_[data[:, :3], np.ones(data[:, :3].shape[0])]
      transf = transf @ rotate_y(90)
      data[:, :3] = transf[:, :-1]

      grp = hf.create_group(str(i))
      grp.create_dataset("img", data=img, compression="gzip")
      grp.create_dataset("points", data=data[:, :3], compression="gzip")
      grp.create_dataset("normals", data=data[:, 3:], compression="gzip", )
      grp.attrs["label"] = y[i]


if __name__ == '__main__':

  with gzip.open('exclude/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="iso-8859-1")

  N_VALID = 100
  for split, set_ in zip((SplitEnum.training,
                          SplitEnum.validation,
                          SplitEnum.testing), (train_set,
                                               valid_set,
                                               test_set)):
    save_dataset(set_[0][:N_VALID],
                 set_[1][:N_VALID],
                 make_voxel(),
                 Path('exclude') / split.value)
