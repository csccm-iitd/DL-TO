import os
import shutil
from itertools import groupby
from pathlib import Path
from zipfile import ZipFile
from tkinter import Tcl

from PIL import Image
from matplotlib import image
import torch
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

ntrain = 10000


def extract_data_from_zip(zip_path):
    with ZipFile(zip_path, 'r') as zip:
        # extracting all the files
        print(f'Extracting files from: {zip_path}')
        zip.extractall(zip_path.parent / "extracted_data")

    return zip_path.parent / "extracted_data"


def create_data_from_images(image_dir_path):
    samples = []
    files = Tcl().call('lsort', '-dict', list(os.walk(image_dir_path))[0][2])


    if "y_train" in str(image_dir_path):
        for i in range(0,len(files),4):
            prefix_arr = []
            for j in range(4):

                image_path = image_dir_path / files[i+j]
                img_arr_rgb = image.imread(image_path)
                if len(img_arr_rgb.shape) != 2: # or = 3
                    img_arr_grey_scale = rgb2gray(img_arr_rgb)
                else:
                    img_arr_grey_scale = img_arr_rgb
                prefix_arr.append(img_arr_grey_scale)
            sample = np.array(prefix_arr)
            samples.append(sample)
    else:
        for file in files:
            image_path = image_dir_path / file
            img_arr_rgb = image.imread(image_path)
            if len(img_arr_rgb.shape) != 2: # or = 3
                img_arr = rgb2gray(img_arr_rgb)
            else:
                img_arr =  img_arr_rgb
            samples.append(img_arr)
    return samples


def rgb2gray(rgb):
    return np.dot(rgb[:, :, :3], [0.2989, 0.5870, 0.1140])


def check_data(data, shape):
    for d in data:
        assert d.shape == shape, "Data shape is invalid"


def load_data(x_dim: int, y_dim: int):

    x_name = f'x_train_new_{x_dim}'
    y_name = f'y_train_{y_dim}'

    data_dir_prefix = Path(__file__).parent.parent / "data" / 'synthetic'
    x_image_dir_path = extract_data_from_zip((data_dir_prefix / x_name).with_suffix(".zip"))
    X = create_data_from_images(x_image_dir_path / x_name)
    y_image_dir_path = extract_data_from_zip((data_dir_prefix / y_name).with_suffix(".zip"))
    y = create_data_from_images(y_image_dir_path / y_name)
    check_data(X, (x_dim, x_dim))
    check_data(y, (4, y_dim, y_dim))
    print(f"X shape: {X[0].shape}")
    print(f"y shape: {y[0].shape}")
    print(f"Total sets of images: {len(X)}")
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)), batch_size=16,
                              shuffle=True, drop_last=True)

    test_loader = DataLoader(TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test)), batch_size=16,
                             shuffle=False, drop_last=True)
    test_loader_nll = DataLoader(TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test)),
                                 batch_size=128, shuffle=False, drop_last=True)
    sample_loader = DataLoader(TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test)),
                               batch_size=1, shuffle=False, drop_last=True)

    return train_loader, test_loader, sample_loader, test_loader_nll, X, y
