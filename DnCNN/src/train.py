import time
import logging
import os
import torch
from torch import nn
import random
from data import *
from model import *
# import evaluate
import myparser
from torch.utils.data.dataloader import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


parser_ = myparser.get_parser("DnCNN")
# noise
parser_.add_argument("--noise_level", type=float, default=15)
parser_.add_argument("--noise_mode", type=str, default="S", help="known noise level(S) or blind(B)")
parser_.add_argument("--noise_level_val", type=float, default=15)
parser_.add_argument("--noise_level_max", type=float, default=55)

args = parser_.parse_args()


class Trainer:
    def __init__(self, dataloader, net, optimizer, loss_fn, logger, scheduler=None):
        self.dataloader = dataloader
        self.net = net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = logger
        self.scheduler = scheduler

    def train(self, index):
        self.logger.info("-----Round{:3d} training begins-----".format(index + 1))
        self.logger.info("lr = {:.5f}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        count_iter = 0
        self.net.train()
        start_time = time.time()
        for _img in self.dataloader:
            self.optimizer.zero_grad()
            _img = _img.to(args.device)
            # _img_noise = _img_noise.to(args.device)
            if args.noise_mode == "S":
                _noise = torch.FloatTensor(_img.shape).normal_(mean=0, std=args.noise_level/255.).to(args.device)
            else:
                _noise = torch.zeros(_img.shape).to(args.device)
                _n_shape = _img[0].shape
                for j in range(_img.shape[0]):
                    _noise_level = random.uniform(0., args.noise_level_max)
                    _noise[j] = torch.FloatTensor(_n_shape).normal_(mean=0, std=_noise_level/255.).to(args.device)

            _img_noise = _img + _noise
            # show_tensor(_img[0])
            # show_tensor(_img_noise[0])
            # break
            _output = self.net(_img_noise)
            _loss = self.loss_fn(_output, _img_noise - _img) / (_img.shape[0] * 2)
            _loss.backward()
            self.optimizer.step()

            count_iter += 1
            if count_iter % 100 == 0:
                end_time = time.time()
                self.logger.info("iter {:4d} finished, loss = {:.10f}, cost time = {:5.2f}"
                                 .format(count_iter, _loss.item(), end_time - start_time))
                start_time = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    def eval(self):
        pass


def main():
    if args.noise_mode == "S":
        model_name = "DnCNN_S_" + str(args.noise_level)
    else:
        model_name = "DnCNN_B"

    if os.path.exists("../logging") == 0:
        os.mkdir("../logging")

    logger = logging.getLogger()
    filehandler = logging.FileHandler(filename="../logging/logging_{:s}.txt".format(model_name), mode="w")
    # streamhandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    filehandler.setFormatter(formatter)
    # streamhandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    # logger.addHandler(streamhandler)
    logger.setLevel('DEBUG')

    # logging.basicConfig(filename="../logging/logging_{:s}.txt".format(model_name), filemode="w", level=logging.DEBUG,
    #                     format='%(asctime)s - %(message)s')
    # # print("Loading Dataset")
    # logging.info("Start Loading Dataset")
    path_train = "../dataset/train/test_{:03d}.png"
    # path_valid = "../dataset/DnCNN_S_valid.h5"
    dataset_train = MyDataset(path_train, 400, int(128*1600/400), 40, logger)
    # dataset_valid = MyDataset(path_valid)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    # loader_valid = DataLoader(dataset_valid, batch_size=7, shuffle=True)

    net = DnCNN(args.num_layers, args.num_channels, args.num_features)
    net.apply(weights_init_kaiming)

    # net = torch.load("../model/DnCNN_S.pth").to(args.device)
    device_id = []
    for i in range(torch.cuda.device_count()):
        device_id.append(i)

    model = nn.DataParallel(net, device_id).to(args.device)
    # net = net.to(args.device)

    loss_fn = nn.MSELoss().to(args.device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.milestone, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.89)
    trainer = Trainer(loader_train, model, optimizer, loss_fn, logger, scheduler)

    for i in range(args.epoch):
        trainer.train(i)
        torch.save(trainer.net, "../model/" + model_name + ".pth")
    max_psnr = 0

    # logging.info("Start Training: {:s}".format(model_name))
    # # print("Start Training, Device = {:s}".format(args.device))
    # for i in range(args.epoch):
    #     logging.info("-----Round{:3d} training begins-----".format(i + 1))
    #     # print("-----Round{:3d} training begins-----".format(i + 1))
    #     logging.info("lr = {:.5f}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    #     # print("lr = {:.5f}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    #     # Train
    #     net.train()
    #     count_iter = 0
    #     start_time = time.time()
    #     for img in loader_train:
    #         # print(img.shape)
    #         if args.noise_mode == "S":
    #             noise = torch.FloatTensor(img.shape).normal_(mean=0, std=args.noise_level/255.).to(args.device)
    #         else:
    #             noise = torch.zeros(img.shape).to(args.device)
    #             n_shape = img[0].shape
    #             for j in range(img.shape[0]):
    #                 noise_level = random.uniform(0., args.noise_level_max)
    #                 noise[j] = torch.FloatTensor(n_shape).normal_(mean=0, std=noise_level/255.).to(args.device)
    #
    #         # print(noise)
    #         # print(noise.shape)
    #         # break
    #         img = img.to(args.device)
    #
    #         img_n = img.clone()
    #         img_n += noise
    #         # print(img)
    #         # print(noise)
    #         # print(img_n)
    #         # time.sleep(10)
    #         # break
    #         # img_n = torch.clamp(img + noise, 0., 1.)
    #         output = net(img_n)
    #         # if count_iter
    #         # print(torch.mean(img), torch.mean(noise), torch.mean(img_n), torch.mean(output))
    #         optimizer.zero_grad()
    #         loss = loss_fn(output, noise) / (img.shape[0] * 2)
    #         # loss1 = loss_fn(img - )
    #         # print(loss.item(), (img.shape[0] * 2))
    #         loss.backward()
    #         optimizer.step()
    #         count_iter += 1
    #         # print(loss.item(), "++")
    #         if count_iter % 100 == 0:
    #             # if count_iter == 3200:
    #             #     src.evaluate.show_tensor(img[0])
    #             #     src.evaluate.show_tensor(img_n[0])
    #             #     src.evaluate.show_tensor(output[0])
    #             # print(src.evaluate.cal_psnr(img[0], img_n[0], torch.Tensor([1]).to(args.device)))
    #             # print(src.evaluate.cal_psnr(img[0], output[0], torch.Tensor([1]).to(args.device)))
    #             end_time = time.time()
    #             logging.info("iter {:4d} finished, loss = {:.10f}, cost time = {:5.2f}".format(count_iter,
    #                                                                                            loss.item(),
    #                                                                                            end_time - start_time))
    #             # print("iter {:4d} finished, loss = {:.10f}, cost time = {:5.2f}".format(count_iter,
    #             #                                                                         loss.item(),
    #             #                                                                         end_time - start_time))
    #             start_time = time.time()
    #
    #     scheduler.step()

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
        # torch.save(net, "../model/" + model_name + ".pth")
        # print("valid_psnr = {:5.2f}".format(total_valid_psnr[0] / len(dataset_valid)))


if __name__ == "__main__":
    main()
