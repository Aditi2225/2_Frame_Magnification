import os
from sklearn.utils import shuffle
import cv2
from skimage.io import imread
from skimage.util import random_noise
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
Tensor = torch.cuda.FloatTensor


def gen_poisson_noise(unit):
    n = np.random.randn(*unit.shape)

    # Strange here, unit has been in range of (-1, 1),
    # but made example to checked to be same as the official codes.
    n_str = np.sqrt(unit + 1.0) / np.sqrt(127.5)
    poisson_noise = np.multiply(n, n_str)
    return poisson_noise


def load_unit(path):
    # Load
    file_suffix = path.split('.')[-1].lower()
    if file_suffix in ['jpg', 'png']:
        try:
            unit = cv2.cvtColor(imread(path).astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print('{} load exception:\n'.format(path), e)
            unit = cv2.cvtColor(np.array(Image.open(path).convert('RGB')), cv2.COLOR_RGB2BGR)
        return unit
    else:
        print('Unsupported file type.')
        return None

def unit_preprocessing(unit, preproc=[], is_test=False):
    # Preprocessing
    if 'BF' in preproc and is_test:
        unit = cv2.bilateralFilter(unit, 9, 75, 75)
    if 'resize' in preproc:
        unit = cv2.resize(unit, (384, 384), interpolation=cv2.INTER_LANCZOS4)
    elif 'downsample' in preproc:
        unit = cv2.resize(unit, unit.shape[1]//2, unit.shape[0]//2, interpolation=cv2.INTER_LANCZOS4)

    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    try:
        if 'poisson' in preproc:
            # Use poisson noise from official repo or skimage?

            # unit = unit + gen_poisson_noise(unit) * np.random.uniform(0, 0.3)

            unit = random_noise(unit, mode='poisson')      # unit: 0 ~ 1
            unit = unit * 255
    except Exception as e:
        print('EX:', e, unit.shape, unit.dtype)

    unit = unit / 127.5 - 1.0

    unit = np.transpose(unit, (2, 0, 1))
    return unit
