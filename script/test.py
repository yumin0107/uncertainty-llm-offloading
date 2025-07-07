import sys
import os
import argparse
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import numpy as np
import torch
import tensorflow as tf

# from datasets
from sionna.phy.channel import RayleighBlockFading

from basestation import BaseStation, User
from user_association import uncertainty_aware_offloading
from model.huggingface_model import HuggingfaceModel

"""
def load_dataset(dataset_dir: str):
"""


def generate_rayleigh_coeffs(N: int, seed: int = None) -> np.ndarray:
    rayleigh = RayleighBlockFading(num_rx=1, num_tx=1, dtype=tf.complex64, seed=seed)
    x = tf.ones((N, 1, 1, 1), dtype=tf.complex64)
    h = rayleigh(x, training=False)
    h = tf.squeeze(h, axis=[1, 2, 3])
    return h.numpy()
