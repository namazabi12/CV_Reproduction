import torch
from torch import nn
import random
from PIL import Image
from torchvision import transforms
from data import *
import myparser
import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr


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


parser_ = myparser.get_parser("DnCNN")
# noise
parser_.add_argument("--noise_level", type=float, default=15)
parser_.add_argument("--noise_mode", type=str, default="S", help="known noise level(S) or blind(B)")
parser_.add_argument("--noise_level_val", type=float, default=15)
parser_.add_argument("--noise_level_max", type=float, default=55)

args = parser_.parse_args()


if __name__ == "__main__":
    model_name = "../model/DnCNN_S_15.pth"
    net = torch.load(model_name)
    print(torch.cuda.device_count())
    # for para in net.parameters():
    #     print(para)
    num_image = 68
    total_psnr = 0
    # dataset_eval = MyDataset("../dataset/train/test_{:03d}.png", num_image, args)
    dataset_eval = MyDataset("../dataset/Set68/test{:03d}.png", num_image, args)
    loader_eval = DataLoader(dataset_eval, batch_size=1)
    # for i in range(1, num_image + 1):
    with torch.no_grad():
        for test_target in loader_eval:
            test_target = test_target.cuda()
            _noise = torch.FloatTensor(test_target.shape).normal_(mean=0, std=args.noise_level / 255.).cuda()
            # test_input = read_image("../datasets/test/2022-11-09_19_38_12_840.bmp").unsqueeze(0).cuda()
            # print(test_input.shape)
            # test_target =  read_image("../dataset/train/test_{:03d}.png".format(i)).unsqueeze(0)
            # test_target =  read_image("../dataset/Set68/test{:03d}.png".format(i)).unsqueeze(0)
            print(test_target.shape)
            # continue
            # test_target = test_target.cuda()
            test_input = test_target + _noise
            # test_input = torch.clamp(test_target + _noise, 0., 1.).cuda()
            # test_target = read_image("../dataset/Set14/image_SRF_2/img_{:03d}_SRF_2_HR.png".format(i)).unsqueeze(0).cuda()
            # noise = torch.FloatTensor(test_target.shape).normal_(mean=0, std=15./255.)
            # test_input = torch.clamp(test_target + noise, 0., 1.)
            # test_input = test_target + noise
            # test_target, test_input = test_target.cuda(), test_input.cuda()

            test_output = test_input - net(test_input)
            loss = nn.MSELoss()(test_output[0], test_target[0]).item()
            show_tensor(test_input[0])
            show_tensor(test_target[0])
            show_tensor(torch.clamp(test_output[0], 0., 1.))
            break
            # print("Image index: {:2d}".format(i))
            # print(cal_psnr(test_input[0], test_output[0], torch.Tensor([1]).cuda()))
            # print("PSNR_output_input  = {:.4f}".format(cal_psnr(test_input[0], test_target[0], torch.Tensor([1]).cuda())[0]))
            psnr = cal_psnr(test_output[0], test_target[0], torch.Tensor([1]).cuda())[0]
            print("PSNR_output_target = {:.4f}".format(psnr))
            print("MSELoss = {:.4f}".format(loss))
            print("loss_psnr{:.4f}".format(10*np.log10(1 / loss)))
            total_psnr += psnr
            # break
        print("Average PSNR = {:.4f}".format(total_psnr / num_image))

    # Image
    # index: 1
    # PSNR_output_target = 30.1850
    # Image
    # index: 2
    # PSNR_output_target = 31.1895
    # Image
    # index: 3
    # PSNR_output_target = 30.7066
    # Image
    # index: 4
    # PSNR_output_target = 29.9361
    # Image
    # index: 5
    # PSNR_output_target = 30.3338
    # Image
    # index: 6
    # PSNR_output_target = 29.8741
    # Image
    # index: 7
    # PSNR_output_target = 30.1225
    # Image
    # index: 8
    # PSNR_output_target = 31.1364
    # Image
    # index: 9
    # PSNR_output_target = 29.2197
    # Image
    # index: 10
    # PSNR_output_target = 30.2940
    # Image
    # index: 11
    # PSNR_output_target = 30.4198
    # Image
    # index: 12
    # PSNR_output_target = 30.1613
    # Average
    # PSNR = 30.2982

