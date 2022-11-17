import torch
from torch import nn
import random
from PIL import Image
from torchvision import transforms
import numpy as np


def read_image(path):
    image = Image.open(path)
    image = transforms.ToTensor()(image)
    return image


def cal_psnr(img0, img1, mx):
    # img0, img1 = img0.astype(np.float64), img1.astype(np.float64)
    mse = torch.mean((img0 - img1) ** 2)
    return 10 * torch.log10(mx ** 2 / mse)
    # return 10 * np.log10(mx ** 2 / mse)


def show_tensor(tensor):
    transforms.ToPILImage()(tensor).show()


if __name__ == "__main__":
    model_name = "../model/DnCNN_S_15.pth"
    net = torch.load(model_name)

    for i in range(1, 13):
        # test_input = read_image("../datasets/test/2022-11-09_19_38_12_840.bmp").unsqueeze(0).cuda()
        # print(test_input.shape)
        test_target =  read_image("../dataset/train/test_{:03d}.png".format(i)).unsqueeze(0)
        # test_target = read_image("../dataset/Set14/image_SRF_2/img_{:03d}_SRF_2_HR.png".format(i)).unsqueeze(0).cuda()
        noise = torch.FloatTensor(test_target.shape).normal_(mean=0, std=15./255.)
        test_input = torch.clamp(test_target + noise, 0., 1.)

        test_target, test_input = test_target.cuda(), test_input.cuda()

        test_output = net(test_input)
        # show_tensor(test_input[0])
        # show_tensor(test_target[0])
        # show_tensor(torch.clamp(test_output[0], 0., 1.))
        print("Image index: {:2d}".format(i))
        # print(cal_psnr(test_input[0], test_output[0], torch.Tensor([1]).cuda()))
        print("PSNR_output_input  = {:.4f}".format(cal_psnr(test_input[0], test_target[0], torch.Tensor([1]).cuda())[0]))
        print("PSNR_output_target = {:.4f}".format(cal_psnr(test_output[0], test_target[0], torch.Tensor([1]).cuda())[0]))
        # break

