# import cv2
from PIL import Image
import numpy as np
import os
import sys
# from multiprocessing import Pool
from os import path as osp
# from tqdm import tqdm

# from basicsr.utils import scandir


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def extract_subimages(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']

    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    for path in img_list:
        img_name, extension = osp.splitext(osp.basename(path))

        # remove the x2, x3, x4 and x8 in the filename for DIV2K
        img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

        # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = np.array(Image.open(path))

        h, w = img.shape[0:2]
        h_space = np.arange(0, h - crop_size + 1, step)
        if h - (h_space[-1] + crop_size) > thresh_size:
            h_space = np.append(h_space, h - crop_size)
        w_space = np.arange(0, w - crop_size + 1, step)
        if w - (w_space[-1] + crop_size) > thresh_size:
            w_space = np.append(w_space, w - crop_size)

        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
                cropped_img = np.ascontiguousarray(cropped_img)
                Image.fromarray(cropped_img).save(osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'),
                                                  quality=70)
                # cv2.imwrite(
                #     osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                #     [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])

opt = {}
opt['n_thread'] = 20
opt['compression_level'] = 1

# HR images
opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_HR'
opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
opt['crop_size'] = 480
opt['step'] = 240
opt['thresh_size'] = 0
extract_subimages(opt)
opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2'
opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
opt['crop_size'] = 240
opt['step'] = 120
opt['thresh_size'] = 0
extract_subimages(opt)

