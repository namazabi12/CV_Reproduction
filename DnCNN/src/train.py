import time

import torch
from torch import nn
import random
from src.data import *
from src.model import *
import src.evaluate
import src.parser
from torch.utils.data.dataloader import DataLoader

parser_ = src.parser.get_parser("DnCNN")
# noise
parser_.add_argument("--noise_level", type=float, default=15)
parser_.add_argument("--noise_mode", type=str, default="B", help="known noise level(S) or blind(B)")
parser_.add_argument("--noise_level_val", type=float, default=15)
parser_.add_argument("--noise_level_max", type=float, default=55)

args = parser_.parse_args()


def main():
    print("Loading Dataset")
    path_train = "../dataset/DnCNN_S_train.h5"
    path_valid = "../dataset/DnCNN_S_valid.h5"
    dataset_train = MyDataset(path_train)
    dataset_valid = MyDataset(path_valid)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    # loader_valid = DataLoader(dataset_valid, batch_size=7, shuffle=True)

    model_name = "DnCNN_" + args.noise_mode
    # net = DnCNN(args.num_layers, args.num_channels, args.num_features).to(args.device)
    net = torch.load("../model/DnCNNDnCNN_B.pth").to(args.device)
    loss_fn = nn.MSELoss().to(args.device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.milestone, gamma=0.1)
    max_psnr = 0

    print("Start Training, Device = {:s}".format(args.device))
    for i in range(args.epoch):
        print("-----Round{:3d} training begins-----".format(i + 1))

        # Train
        net.train()
        count_iter = 0
        start_time = time.time()
        for img in loader_train:
            # print(img.shape)
            if args.noise_mode == "S":
                noise = torch.FloatTensor(img.shape).normal_(mean=0, std=args.noise_level/255.).to(args.device)
            else:
                noise = torch.zeros(img.shape).to(args.device)
                n_shape = img[0].shape
                for j in range(img.shape[0]):
                    noise_level = random.uniform(0., args.noise_level_max)
                    noise[j] = torch.FloatTensor(n_shape).to(args.device).normal_(mean=0, std=noise_level/255.).to(args.device)

            img = img.to(args.device)
            img_n = torch.clamp(img + noise, 0., 1.)
            output = net(img_n)
            # if count_iter
            # print(torch.mean(img), torch.mean(noise), torch.mean(img_n), torch.mean(output))
            optimizer.zero_grad()
            loss = loss_fn(output, img) / (img.shape[0] * 2)
            # print(loss.item(), (img.shape[0] * 2))
            loss.backward()
            optimizer.step()
            count_iter += 1
            # print(loss.item(), "++")
            if count_iter % 100 == 0:
                # if count_iter == 3200:
                #     src.evaluate.show_tensor(img[0])
                #     src.evaluate.show_tensor(img_n[0])
                #     src.evaluate.show_tensor(output[0])
                # print(src.evaluate.cal_psnr(img[0], img_n[0], torch.Tensor([1]).to(args.device)))
                # print(src.evaluate.cal_psnr(img[0], output[0], torch.Tensor([1]).to(args.device)))
                end_time = time.time()
                print("iter {:4d} finished, loss = {:.10f}, cost time = {:5.2f}".format(count_iter,
                                                                                        loss.item(),
                                                                                        end_time - start_time))
                start_time = time.time()

        scheduler.step()

        # Valid
        # net.eval()
        # total_valid_psnr = 0
        # for img in loader_valid:
        #     # if args.noise_mode == "S":
        #     noise = torch.FloatTensor(img.shape).normal_(mean=0, std=15. / 255.)
        #     # else:
        #     #     noise = torch.zeros(img.shape)
        #     #     n_shape = img[0].shape
        #     #     for j in range(img.shape[0]):
        #     #         noise_level = random.uniform(0., args.noise_level_max)
        #     #         noise[j] = torch.FloatTensor(n_shape).normal_(mean=0, std=noise_level / 255.)
        #
        #     img_n = torch.clamp(img + noise, 0., 1.)
        #     img, img_n = img.to(args.device), img_n.to(args.device)
        #     output = net(img_n)
        #     for j in range(img.shape[0]):
        #         psnr1 = src.evaluate.cal_psnr(img[j], img_n[j], torch.Tensor([1]).to(args.device))
        #         psnr2 = src.evaluate.cal_psnr(img[j], output[j], torch.Tensor([1]).to(args.device))
        #         print(psnr1, psnr2)
        #         # loss = loss_fn(output, img_n)
        #         total_valid_psnr += psnr2
        #
        # if total_valid_psnr > max_psnr:
        #     max_psnr = total_valid_psnr
        torch.save(net, "../model/DnCNN" + model_name + ".pth")
        # print("valid_psnr = {:5.2f}".format(total_valid_psnr[0] / len(dataset_valid)))


if __name__ == "__main__":
    main()
