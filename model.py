#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/13

import numpy as np
from numpy import ndarray
import seaborn as sns
import matplotlib.pyplot as plt


def relu(x:ndarray) -> ndarray:
  return x * (x > 0)

def softmax(x:ndarray, t:float=1.0) -> ndarray:
  e_x = np.exp(t * x)
  return e_x / np.sum(e_x, axis=0, keepdims=True)


class Attn:
  
  def __init__(self, Q:ndarray, K:ndarray, V:ndarray):
    self.Q = Q
    self.K = K
    self.V = V

  def __call__(self, x:ndarray) -> ndarray:
    # x: [F=6, 1+N=5, P=2]
    q = np.einsum('df, fnp -> dnp', self.Q, x)
    k = np.einsum('df, fnp -> dnp', self.K, x)  
    c = np.einsum('dnk, dmk -> dnm', k, q)

    # just drop dim D here, since the data is 1-dim :(
    c0 = c[0]
    # the temparature must be large enough to reduce numeric error :(
    sc = softmax(c0, 4)   # [n, m]
    if not 'plot': sns.heatmap(sc) ; plt.show()

    v = np.einsum('df, fnp -> dnp', self.V, x)
    o = np.einsum('dnp, nm -> dmp', v, sc)
    return x + o


class FFW:

  def __init__(self, W1:ndarray, b1:ndarray, W2:ndarray, b2:ndarray):
    self.W1, self.b1 = W1, b1
    self.act = relu
    self.W2, self.b2 = W2, b2

  def __call__(self, x:ndarray) -> ndarray:
    r = x
    # x: [F=6, 1+N=5, P=2]
    breakpoint()
    x = self.W1 @ x + self.b1
    x = self.act(x)
    x = self.W2 @ x + self.b2
    return r + x


class Transformer:

  def __init__(self, attn:Attn, ffw:FFW):
    self.attn = attn
    self.ffw = ffw

  def __call__(self, x:ndarray) -> ndarray:
    x = self.attn(x)
    r = self.ffw(x)
    return x + r
