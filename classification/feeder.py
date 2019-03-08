import multiprocessing as mp
import queue
import time
import traceback
from functools import wraps

import numpy as np
from warg import NOD


class DataFeeder():
  '''
      This is a workaround of Pythons extremely slow interprocess communication pipes.
      The ideal solution would be to use a multiprocessing.queue, but it apparently communication is band
      limited.
      This solution has processes complete tasks (batches) and a thread add the results to a queue.queue.
  '''

  def __init__(self, func, *args, max_size=100, n_proc=3, max_tasks_per_child=3, **kwargs):
    self._max_size = max_size
    self._func = func
    self._args = args
    self._kwargs = kwargs

    self._queue = queue.Queue(maxsize=max_size)
    self._pool = mp.Pool(n_proc, maxtasksperchild=max_tasks_per_child)

    for i in range(max_size):
      self.fill()

  def close(self):
    self._pool.close()
    self._pool.join()

  def terminate(self):
    self._pool.terminate()
    self._pool.join()

  def fill(self):
    if self.queue_size() < self._max_size:
      self._pool.apply_async(self._func, self._args, self._kwargs, self.put)

  def queue_size(self):
    return self._queue.qsize()

  def put(self, *args, **kwargs):
    if isinstance(args[0], Exception):
      raise args[0]
    self._queue.put(*args, **kwargs)

  def get(self, *args, **kwargs):
    res = self._queue.get(*args, **kwargs)
    self.fill()
    return res

def generate_augmented_image(i):
  return (np.zeros((2, 2)), i)
  #return NOD(img=np.zeros((2, 2)), a=i)

@wraps
def a(batch_size = 8):
  def generate_batch():
    try:
      batch = [generate_augmented_image(i) for i in range(batch_size)]
      imgs = np.array([i[0] for i in batch], dtype=np.float32)
      ground_truth = np.array([i[1] for i in batch], dtype=np.float32)
      return (imgs, ground_truth)
    except Exception as inst:
      traceback.print_exc()
      return inst
  return generate_batch

if __name__ == '__main__':
  t1 = time.time()
  df = DataFeeder(a, n_proc=8, max_tasks_per_child=10)
  t2 = time.time()
  batch = df.get()
  t3 =  time.time()

  a = t2 - t1
  b = t3 - t2

  print(batch,a,b)
