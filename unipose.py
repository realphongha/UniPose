# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import cv2
import math
import os

sys.path.append("..")
from unipose_utils.utils import get_model_summary
from unipose_utils.utils import adjust_learning_rate as adjust_learning_rate
from unipose_utils.utils import save_checkpoint      as save_checkpoint
from unipose_utils.utils import printAccuracies      as printAccuracies
from unipose_utils.utils import guassian_kernel      as guassian_kernel
from unipose_utils.utils import get_parameters       as get_parameters
from unipose_utils       import Mytransforms         as  Mytransforms 
from unipose_utils.utils import getDataloader        as getDataloader
from unipose_utils.utils import getOutImages         as getOutImages
from unipose_utils.utils import AverageMeter         as AverageMeter
from unipose_utils.utils import draw_paint           as draw_paint
from unipose_utils       import evaluate             as evaluate
from unipose_utils.utils import get_kpts             as get_kpts
from unipose_utils.utils import draw_paint_simple    as draw_paint_simple

from model.unipose import unipose

from tqdm import tqdm

import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

from PIL import Image


def get_model(ckpt="checkpoint_best.pth.tar", cpu=True):
    model = unipose("MPII", num_classes=16, backbone='resnet', 
                    output_stride=16,
                    sync_bn=True, freeze_bn=False, stride=8)
    print("Loading checkpoint...")
    if cpu:
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(ckpt)
    if "state_dict" in checkpoint:
        p = checkpoint['state_dict']
    else:
        p = checkpoint

    state_dict = model.state_dict()
    model_dict = {}

    for k,v in p.items():
        if k in state_dict:
            model_dict[k] = v

    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    print("Loaded checkpoint!")
    model.eval()
    return model


def detect(image, model, cpu=True):
    model.eval()
    ori_img = cv2.resize(image, (368, 368))
    img = np.array(ori_img, dtype=np.float32)
    center   = [184, 184]
    img  = img.transpose(2, 0, 1)
    img  = torch.from_numpy(img)
    mean = [128.0, 128.0, 128.0]
    std  = [256.0, 256.0, 256.0]
    for t, m, s in zip(img, mean, std):
        t.sub_(m).div_(s)

    img = torch.unsqueeze(img, 0)

    input_var = img.cuda() if not cpu else img

    heat = model(input_var)

    heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

    kpts = get_kpts(heat, img_h=368.0, img_w=368.0)
    ori_img = draw_paint_simple(ori_img, kpts, "MPII")

    return ori_img


class Trainer(object):
    def __init__(self, args):
        self.args         = args
        self.train_dir    = args.train_dir
        self.val_dir      = args.val_dir
        self.model_arch   = args.model_arch
        self.dataset      = args.dataset


        self.workers      = 4
        self.weight_decay = 0.0005
        self.momentum     = 0.9
        self.batch_size   = 8
        self.lr           = 0.0001
        self.gamma        = 0.333
        self.step_size    = 13275
        self.sigma        = 3
        self.stride       = 8

        cudnn.benchmark   = True

        if self.dataset   ==  "LSP":
            self.numClasses  = 14
        elif self.dataset == "MPII":
            self.numClasses  = 16
        
        if args.mode != "test":
            self.train_loader, self.val_loader, _ = getDataloader(self.dataset, self.train_dir, self.val_dir,
                self.val_dir, self.sigma, self.stride, self.workers, self.batch_size)

        model = unipose(self.dataset, num_classes=self.numClasses,backbone='resnet',output_stride=16,sync_bn=True,freeze_bn=False, stride=self.stride)

        self.model       = model.cuda() if not self.args.cpu else model

        self.criterion   = nn.MSELoss().cuda() if not self.args.cpu else nn.MSELoss()

        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.best_model  = 12345678.9

        self.iters       = 0

        if self.args.pretrained is not None:
            print("Loading checkpoint...")
            if args.cpu:
                checkpoint = torch.load(self.args.pretrained, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(self.args.pretrained)
            if "state_dict" in checkpoint:
                p = checkpoint['state_dict']
            else:
                p = checkpoint

            state_dict = self.model.state_dict()
            model_dict = {}

            for k,v in p.items():
                if k in state_dict:
                    model_dict[k] = v

            state_dict.update(model_dict)
            try:
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print("Exception:", e)
                state_dict = self.model.state_dict()
                model_dict = {}

                for k,v in p.items():
                    if k in state_dict and "decoder" not in k:
                        model_dict[k] = v
                state_dict.update(model_dict)
                self.model.load_state_dict(state_dict, strict=False)
            print("Loaded checkpoint!")
            
        self.isBest = 0
        self.bestPCK  = 0
        self.bestPCKh = 0

        # Print model summary and metrics
        if args.cpu:
            dump_input = torch.rand((1, 3, 368, 368))
        else:
            dump_input = torch.rand((1, 3, 368, 368)).cuda()
        print(get_model_summary(self.model, dump_input))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        print("Epoch " + str(epoch) + ':') 
        tbar = tqdm(self.train_loader)

        for i, (input, heatmap, centermap, img_path) in enumerate(tbar):
            learning_rate = adjust_learning_rate(self.optimizer, self.iters, self.lr, policy='step',
                                                 gamma=self.gamma, step_size=self.step_size)

            input_var     =     input.cuda() if not self.args.cpu else input
            heatmap_var   =    heatmap.cuda() if not self.args.cpu else heatmap

            self.optimizer.zero_grad()

            heat = self.model(input_var)

            loss_heat   = self.criterion(heat,  heatmap_var)

            loss = loss_heat

            train_loss += loss_heat.item()

            loss.backward()
            self.optimizer.step()

            tbar.set_description('Train loss: %.6f' % (train_loss / ((i + 1)*self.batch_size)))

            self.iters += 1

            if i == 10000:
            	break

    def validation(self, epoch, save=True):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        
        AP    = np.zeros(self.numClasses+1)
        PCK   = np.zeros(self.numClasses+1)
        PCKh  = np.zeros(self.numClasses+1)
        count = np.zeros(self.numClasses+1)

        cnt = 0
        for i, (input, heatmap, centermap, img_path) in enumerate(tbar):

            cnt += 1

            input_var     =      input.cuda() if not self.args.cpu else input
            heatmap_var   =    heatmap.cuda() if not self.args.cpu else heatmap
            self.optimizer.zero_grad()

            heat = self.model(input_var)
            loss_heat   = self.criterion(heat,  heatmap_var)

            loss = loss_heat

            val_loss += loss_heat.item()

            tbar.set_description('Val   loss: %.6f' % (val_loss / ((i + 1)*self.batch_size)))

            acc, acc_PCK, acc_PCKh, cnt, pred, visible = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),0.2,0.5, self.dataset)

            AP[0]     = (AP[0]  *i + acc[0])      / (i + 1)
            PCK[0]    = (PCK[0] *i + acc_PCK[0])  / (i + 1)
            PCKh[0]   = (PCKh[0]*i + acc_PCKh[0]) / (i + 1)

            for j in range(1,self.numClasses+1):
                if visible[j] == 1:
                    AP[j]     = (AP[j]  *count[j] + acc[j])      / (count[j] + 1)
                    PCK[j]    = (PCK[j] *count[j] + acc_PCK[j])  / (count[j] + 1)
                    PCKh[j]   = (PCKh[j]*count[j] + acc_PCKh[j]) / (count[j] + 1)
                    count[j] += 1

            mAP     =   AP[1:].sum()/(self.numClasses)
            mPCK    =  PCK[1:].sum()/(self.numClasses)
            mPCKh   = PCKh[1:].sum()/(self.numClasses)
	
        printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, self.dataset)
            
        PCKhAvg = PCKh.sum()/(self.numClasses+1)
        PCKAvg  =  PCK.sum()/(self.numClasses+1)

        if mAP > self.isBest:
            self.isBest = mAP
            if save:
                if not self.args.model_name:
                    self.args.model_name = "checkpoint"
                save_checkpoint({'state_dict': self.model.state_dict()}, self.isBest, self.args.save_path, self.args.model_name)
                print("Model saved to "+self.args.model_name)

        if mPCKh > self.bestPCKh:
            self.bestPCKh = mPCKh
        if mPCK > self.bestPCK:
            self.bestPCK = mPCK

        print("Best AP = %.2f%%; PCK = %2.2f%%; PCKh = %2.2f%%" % (self.isBest*100, self.bestPCK*100,self.bestPCKh*100))



    def test(self,epoch):
        self.model.eval()
        print("Testing...") 
        img_path = args.test_img
        img_ori  = np.array(cv2.resize(cv2.imread(img_path),(368,368)), dtype=np.float32)
        img_ori2 = cv2.resize(cv2.imread(img_path),(368,368))

        no_samples = 1
        for idx in range(no_samples):
            print(idx+1, "/", no_samples)

            center   = [184, 184]

            img = img_ori.copy()
            img  = img.transpose(2, 0, 1)
            img  = torch.from_numpy(img)
            mean = [128.0, 128.0, 128.0]
            std  = [256.0, 256.0, 256.0]
            for t, m, s in zip(img, mean, std):
                t.sub_(m).div_(s)

            img       = torch.unsqueeze(img, 0)

            self.model.eval()

            input_var   = img.cuda() if not self.args.cpu else img

            heat = self.model(input_var)

            heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

            kpts = get_kpts(heat, img_h=368.0, img_w=368.0)
            draw_paint(img_path, kpts, idx, epoch, self.model_arch, self.dataset)

            heat = heat.detach().cpu().numpy()

            heat = heat[0].transpose(1,2,0)


            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    for k in range(heat.shape[2]):
                        if heat[i,j,k] < 0:
                            heat[i,j,k] = 0
                        
            im = img_ori2.copy()

            heatmap = []
            for i in range(self.numClasses+1):
                heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
                im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
                cv2.imwrite('samples/heat/unipose' + str(i) + '.png', im_heat)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None,type=str, dest='pretrained')
    parser.add_argument('--dataset', type=str, dest='dataset', default='LSP')
    parser.add_argument('--train_dir', default='/PATH/TO/TRAIN',type=str, dest='train_dir')
    parser.add_argument('--val_dir', type=str, dest='val_dir', default='/PATH/TO/LSP/VAL')
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--model_arch', default='unipose', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--test_img', default='pose.jpeg', type=str)
    parser.add_argument('--save_path', default='.', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--cpu', action='store_true', default=False, help='Use CPU instead of GPU or not?')
    args = parser.parse_args()

    starter_epoch =    0
    epochs        =  args.epoch

    args = parser.parse_args()

    if not args.gpu.isdigit():
        print("%s is not a valid GPU number" % args.gpu)
        quit()

    if not args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("Start training with config:")
    print(args)

    trainer = Trainer(args)

    if args.mode.lower() == "val":
        trainer.validation(1, save=False)
        quit()

    if args.mode.lower() == "train" or args.mode.lower() == "both":
        for epoch in range(starter_epoch, epochs):
            trainer.training(epoch)
            trainer.validation(epoch)
        
    if args.mode.lower() == "test" or args.mode.lower() == "both":
        trainer.test(0)
