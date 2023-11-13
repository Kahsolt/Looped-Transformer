#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/13

from model import *


class Read:

  def __init__(self):
    K = Q = np.asarray([[0, 0, 1, 1, 0, 0]])  # [D=1, F=6]
    V = np.asarray([
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
    ])
    W1 = np.asarray([
      [1,  0, 0, 0, 0, 0],
      [0, -2, 0, 0, 2, 0],      # 2*(v_new-v_orig)
      [0,  0, 1, 0, 0, 0],
      [0,  0, 0, 1, 0, 0],
      [0,  0, 0, 0, 1, 0],
      [0,  0, 0, 0, 0, 1],
    ])
    C = 100   # arge postive constant
    b1 = np.asarray([0, -1, -1, -1, -1, -1]) * C    # C(b-1)
    W2 = np.asarray([
      # what the fxck?
      # FIXME: how dare you
    ])
    b2 = np.asarray([0, -1, -1, -1, -1, -1]) * C
    self.tf = Transformer(
      Attn(K, Q, V),
      FFW(W1, b1, W2, b2),
    )

  def __call__(self, x:ndarray) -> ndarray:
    print(x.shape)
    return self.tf(x)


if __name__ == '__main__':
  # place holders
  X = -7
  _ = -1
  # input schema (1d-data seq for demo)
  x = np.asarray([             # [1+N, F=6]
    # pad |    mem   | cmd
    [  0,  1, 2, 3, 4     ],   # data seq
    [  X,  0, 0, 0, 0     ],   # tgt data pos enc
    [  _,  0, 0, 0, 0     ],   # tgt pos enc
    [  0,  _, _, _, _     ],   # data pos enc
    [  0,  0, 0, 0, 0     ],   # tmp storage (NOTE: tricky!)
    [  1,  0, 0, 0, 0     ],   # flag for pad area
  ])

  # data seqlen
  N = x.shape[1] - 1
  # log(N) posenc depth  
  logN = int(np.ceil(np.log(N)))
  # x_exp, [1+N, F=6, logN=2]
  x = np.tile(np.expand_dims(x, axis=-1), reps=(1, logN))
  # assign pos enc
  for i in range(N):
    posbin = np.asarray([int(e) for e in bin(i)[2:].rjust(logN, '0')])
    posenc = posbin * 2 - 1   # vrng ±1
    x[3, 1+i, :] = posenc
  # assign tgt pos enc
  tgt_idx = 2       # the value idx to read, NOTE: use can change this~
  x[2, 0, :] = x[3, 1+tgt_idx, :]
  # preparation done
  print('x:', x)

  read = Read()
  o = read(x)
  print('o', o)
