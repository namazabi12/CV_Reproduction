import random
import numpy as np
import torch
from torch import nn
import math
# import h5py
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def read_image(path):
    image = Image.open(path)
    # image = np.array(image) / 255.
    image = transforms.ToTensor()(image)
    return image


def show_tensor(tensor):
    transforms.ToPILImage()(tensor).show()


# def color_channel_change(img):


# def data_prepare(data_dir: str, num_img, num_patch, crop_size, save_name: str):
#
#     # h5f = h5py.File(save_name + ".h5", 'w')
#     # h5f.clear()
#     num_save = 0
#
#     if crop_size:
#         for i in range(1, num_img + 1):
#             if i % 100 == 0:
#                 print("image{:04d} complete".format(i))
#             _img = read_image(data_dir.format(i))
#             for j in range(num_patch):
#                 img_patch = transforms.RandomCrop(crop_size)(_img)
#                 # h5f.create_dataset(str(num_save), data=np.array(img_patch))
#                 num_save += 1
#     else:
#         for i in range(1, num_img + 1):
#             _img = read_image(data_dir.format(i))
#             # h5f.create_dataset(str(num_save), data=np.array(_img))
#             num_save += 1
#
#     # h5f.close()


class MyDataset(Dataset):
    def __init__(self, data_dir: str, num_img, num_patch=None, crop_size=None, logger=None):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.num_img = num_img
        self.num_patch = num_patch
        self.crop_size = crop_size
        self.logger = logger

        self.data = []
        self.data_noise = []

        if self.crop_size:
            if type(self.crop_size) == int:
                self.crop_size = [self.crop_size, self.crop_size]
            for i in range(self.num_img):
                if (i + 1) % 100 == 0:
                    # print("image{:04d} complete".format(i + 1))
                    if self.logger is not None:
                        self.logger.info("Dataset: image{:04d} completed".format(i + 1))
                self.img = read_image(self.data_dir.format(i + 1))
                self.c, self.h, self.w = self.img.shape
                for j in range(self.num_patch):
                    self.ch = random.randint(0, self.h - self.crop_size[0])
                    self.cw = random.randint(0, self.w - self.crop_size[1])
                    self.data.append(self.img[:, self.ch:self.ch+self.crop_size[0], self.cw:self.cw+self.crop_size[1]])
        else:
            for i in range(1, self.num_img + 1):
                self.img = read_image(self.data_dir.format(i))
                self.data.append(self.img)

        # for _img in self.data:
        #     noise = torch.FloatTensor(_img.shape).normal_(mean=0, std=15./255.)
        #     self.data_noise.append(_img + noise)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    # pass
    path_train = "../dataset/train/test_{:03d}.png"
    path_valid = "../dataset/Set12/{:02d}.png"
    save_train = "../dataset/DnCNN_S_train"
    save_valid = "../dataset/DnCNN_S_valid"

    # data_prepare(path_train, 400, int(128*1600/400), 40, save_train)
    # data_prepare(path_valid, 12, 1, 0, save_valid)
    #
    md_train = MyDataset(path_train, 400, 1, 40)
    loader_train = DataLoader(md_train, batch_size=32, shuffle=True)
    for img in loader_train:
        print(img.shape)
        show_tensor(img[0])
    # md_valid = MyDataset(save_valid + ".h5")
    #
    # print(len(md_train))
    # print(len(md_valid))
    # for img in md_train:
    #     show_tensor(img)


