# -*-coding:UTF-8-*-
from __future__ import print_function, absolute_import

import json
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data

import unipose_utils.Mytransforms as Mytransforms


def get_transform(center, scale, resolution):
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(resolution[1]) / h
    t[1, 1] = float(resolution[0]) / h
    t[0, 2] = resolution[1] * (-float(center[0]) / h + .5)
    t[1, 2] = resolution[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1

    return t


def transformImage(pt, center, scale, resolution):
    t = get_transform(center, scale, resolution)
    t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)

    return new_pt[:2].astype(int) + 1


def crop(img, points, center, scale, resolution):
    upperLeft = np.array(transformImage([0, 0], center, scale, resolution))
    bottomRight = np.array(transformImage(resolution, center, scale, resolution))

    # Range to fill new array
    new_x = max(0, -upperLeft[0]), min(bottomRight[0], img.shape[1]) - upperLeft[0]
    new_y = max(0, -upperLeft[1]), min(bottomRight[1], img.shape[0]) - upperLeft[1]
    # Range to sample from original image
    old_x = max(0, upperLeft[0]), min(img.shape[1], bottomRight[0])
    old_y = max(0, upperLeft[1]), min(img.shape[0], bottomRight[1])
    new_img = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    points[:, 0] = points[:, 0] - max(0, upperLeft[0])  # + max(0, -upperLeft[0])
    points[:, 1] = points[:, 1] - max(0, upperLeft[1])  # + max(0, -upperLeft[1])

    center[0] -= max(0, upperLeft[0])
    center[1] -= max(0, upperLeft[1])

    return new_img, upperLeft, bottomRight, points, center


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class mpii(data.Dataset):
    def __init__(self, root_dir, sigma, is_train, single_person=True, transform=None):
        self.width = 368
        self.height = 368
        self.transformer = transform
        self.is_train = is_train
        self.sigma = sigma
        self.parts_num = 16
        self.stride = 8

        self.labels_dir = root_dir
        self.images_dir = root_dir + 'images/'

        self.videosFolders = {}
        self.labelFiles = {}
        self.full_img_list = {}
        self.numPeople = []
        self.single_person = single_person

        with open(self.labels_dir + "mpii_%s.json" % (self.is_train.lower())) as anno_file:
            self.anno = json.load(anno_file)

        with open(self.labels_dir + "annot_mpii.json") as anno_file:
            file = json.load(anno_file)
            files = file["images"]
            bbox = file["annotations"]
        filename_mapping = dict()
        for fn in files:
            filename_mapping[fn["id"]] = fn["file_name"]
 
        self.bbox_mapping = dict()
        for bb in bbox:
            if filename_mapping[bb["image_id"]] not in self.bbox_mapping:
                self.bbox_mapping[filename_mapping[bb["image_id"]]] = list()
            new_bb = (bb["bbox"][0], bb["bbox"][1], bb["bbox"][0]+bb["bbox"][2], bb["bbox"][1]+bb["bbox"][3])
            self.bbox_mapping[filename_mapping[bb["image_id"]]].append(new_bb)

        # print(self.bbox_mapping);quit()
        # for fn in self.bbox_mapping:
        #     img = cv2.imread(self.images_dir + fn)
        #     for bb in self.bbox_mapping[fn]:
        #         print(bb, bb[:-2], bb[-2:])
        #         img = cv2.rectangle(img, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (255, 0, 0), 2)
        #     cv2.imwrite(fn, img)
        #     quit()

        self.img_list = []
        self.new_anno = []

        for val in self.anno:
            if os.path.isfile(self.images_dir + val['image']) and val["image"] in self.bbox_mapping:
                self.new_anno.append(val)

        self.anno = self.new_anno
        
        for idx, val in enumerate(self.anno):
            self.img_list.append(idx) 

        print("No. images:", len(self.img_list))

    def __getitem__(self, index):
        scale_factor = 0.25

        variable = self.anno[self.img_list[index]]

        # while not os.path.isfile(self.images_dir + variable['image']) or variable["image"] not in self.bbox_mapping:
        #     index = index - 1
        #     variable = self.anno[self.img_list[index]]

        img_path = self.images_dir + variable['image']

        # BBox was added to the labels by the authors to perform additional training and testing, as referred in the paper.
        # Intentionally left as comment since it is not part of the dataset.
        #         bbox      = np.load(self.labels_dir + "BBOX/" + variable['img_paths'][:-4] + '.npy')

        points = torch.Tensor(variable['joints'])
        center = torch.Tensor(variable['center'])
        scale = variable['scale']

        if center[0] != -1:
            center[1] = center[1] + 15 * scale
            scale = scale * 1.25

        # Single Person
        nParts = points.size(0)
        img = cv2.imread(img_path)
        if self.single_person:
            bbox = np.array(self.bbox_mapping[variable["image"]])
            box = np.zeros((2,2))

            for i in range(bbox.shape[0]):
                if center[0] > bbox[i,0] and center[0] < bbox[i,2] and\
                    center[1] > bbox[i,1] and center[1] < bbox[i,3]:

                    upperLeft   = bbox[i,0:2].astype(int)
                    bottomRight = bbox[i,-2:].astype(int)
                    box = bbox[i,:]

                    img[:,0:upperLeft[0],:]  = np.ones(img[:,0:upperLeft[0],:].shape) *255
                    img[0:upperLeft[1],:,:]  = np.ones(img[0:upperLeft[1],:,:].shape) *255
                    img[:,bottomRight[0]:,:] = np.ones(img[:,bottomRight[0]:,:].shape)*255
                    img[bottomRight[1]:,:,:] = np.ones(img[bottomRight[1]:,:,:].shape)*255

                    break

            img, upperLeft, bottomRight, points, center = crop(img, points, center, scale, [self.height, self.width])

        kpt = points

        # img, kpt, center = self.transformer(img, points, center)
        if img.shape[0] != 368 or img.shape[1] != 368:
            kpt[:, 0] = kpt[:, 0] * (368 / img.shape[1])
            kpt[:, 1] = kpt[:, 1] * (368 / img.shape[0])
            img = cv2.resize(img, (368, 368))
        height, width, _ = img.shape

        heatmap = np.zeros((int(height / self.stride), int(width / self.stride), int(len(kpt) + 1)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=int(height / self.stride), size_w=int(width / self.stride), center_x=x,
                                       center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

        centermap = np.zeros((int(height / self.stride), int(width / self.stride), 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=int(height / self.stride), size_w=int(width / self.stride),
                                     center_x=int(center[0] / self.stride), center_y=int(center[1] / self.stride),
                                     sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        # orig_img = img.copy()
        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                     [256.0, 256.0, 256.0])
        heatmap = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(centermap)

        # return img, heatmap, centermap, img_path, orig_img
        return img, heatmap, centermap, img_path

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    ds = mpii("/mnt/hdd10tb/Users/phonghh/data/pose/MPII/", 1, "Train")
    for i in range(200, 240):
        img, heatmap, centermap, img_path, orig_img = ds[i]
        img_name = img_path.split("/")[-1].split(".")[0]
        image = cv2.imread(img_path)
        # print(centermap.shape)
        # print(centermap[0].shape)
        # print(centermap)
        # cv2.imwrite(img_name + "_centermap.jpg", centermap[0].numpy())
        # print(img.shape)
        # print(img)
        # quit()
        # img = img.numpy()
        # img = img.transpose(1, 2, 0)
        # cv2.imwrite(img_name + "_ori.jpg", cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA))
        # cv2.imwrite(img_name + ".jpg", cv2.resize(orig_img, (512, 512), interpolation=cv2.INTER_AREA))
        print(heatmap)
        print(heatmap.shape)
        heatmap = heatmap.sum(axis=0)
        heatmap /= heatmap.max()
        # print(heatmap)
        # print(heatmap.shape)
        # cv2.imwrite(img_name + "_heatmap.jpg", cv2.resize(heatmap.numpy(), (512, 512), interpolation=cv2.INTER_AREA))
