import random
import numpy as np
import torch
import h5py
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def read_image(path):
    image = Image.open(path)
    # image = np.array(image) / 255.
    return image


def show_tensor(tensor):
    transforms.ToPILImage()(tensor).show()


def data_prepare(data_dir: str, num_img, num_patch, crop_size, save_name: str):

    h5f = h5py.File(save_name + ".h5", 'w')
    h5f.clear()
    num_save = 0

    if crop_size:
        for i in range(1, num_img + 1):
            if i % 100 == 0:
                print("image{:04d} complete".format(i))
            _img = read_image(data_dir.format(i))
            for j in range(num_patch):
                img_patch = transforms.RandomCrop(crop_size)(_img)
                h5f.create_dataset(str(num_save), data=np.array(img_patch))
                num_save += 1
    else:
        for i in range(1, num_img + 1):
            _img = read_image(data_dir.format(i))
            h5f.create_dataset(str(num_save), data=np.array(_img))
            num_save += 1

    h5f.close()


class MyDataset(Dataset):
    def __init__(self, data_dir):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        h5f = h5py.File(self.data_dir, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.data_dir, 'r')
        key = self.keys[index]
        data = h5f[key][:]
        h5f.close()
        return transforms.ToTensor()(data)


if __name__ == "__main__":
    # pass
    path_train = "../dataset/train/test_{:03d}.png"
    path_valid = "../dataset/Set12/{:02d}.png"
    save_train = "../dataset/DnCNN_S_train"
    save_valid = "../dataset/DnCNN_S_valid"

    data_prepare(path_train, 400, int(128*1600/400), 40, save_train)
    # data_prepare(path_valid, 12, 1, 0, save_valid)
    #
    md_train = MyDataset(save_train + ".h5")
    # md_valid = MyDataset(save_valid + ".h5")
    #
    # print(len(md_train))
    # print(len(md_valid))
    # for img in md_train:
    #     show_tensor(img)


