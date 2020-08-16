import os
import numpy as np
import logging
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as ttransforms
from . import data_transforms_bp_yolo as data_transforms


class COCODataset(Dataset):
    def __init__(self, list_path, img_size, is_training, is_debug=False):
        self.img_files = []
        self.label_files = []
        self.label_coor_files = []
        self.labelh_files = []
        for path in open(list_path, 'r'):
            label_path = path.replace('flickr', 'flickr_heatmap').replace('PLVP', 'PLVP_heatmap').replace('URVP', 'URVP_heatmap').replace('kong', 'kong_heatmap').replace('linepoint', 'lineheatmap').strip()#for training
            labelh_path = path.replace('flickr', 'flickr_heatmap_h').replace('PLVP', 'PLVP_heatmap_h').replace('URVP', 'URVP_heatmap_h').replace('kong', 'kong_heatmap_h').replace('linepoint', 'lineheatmap_h').strip()
            coord_path = path.replace('flickr', 'flickrtx').replace('PLVP', 'PLVPtx').replace('kong', 'kongtx').replace('.png', '.txt').replace('.jpg', '.txt').strip()

            if os.path.isfile(label_path):
                self.img_files.append(path)
                self.label_files.append(label_path)
                self.label_coor_files.append(coord_path)
                self.labelh_files.append(labelh_path)
            else:
                logging.info("no label found. skip it: {}".format(label_path))
        logging.info("Total images: {}".format(len(self.img_files)))
        self.img_size = img_size  # (w, h)
        # print(img_size[0])
        self.max_objects = 5
        self.is_debug = is_debug

        #  transforms and augmentation
        self.transforms = data_transforms.Compose()
        if is_training:
            self.transforms.add(data_transforms.ImageBaseAug())
        # self.transforms.add(data_transforms.KeepAspect())
        self.transforms.add(data_transforms.ResizeImage(self.img_size))
        self.transforms.add(data_transforms.ToTensor(self.max_objects, self.is_debug))

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img / 255.

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labelh_path = self.labelh_files[index % len(self.img_files)].rstrip()
        coord_path = self.label_coor_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            # print(label_path)
            labels = np.loadtxt(coord_path).reshape(-1, 5)
            line = cv2.imread(label_path, 0)
            line_h = cv2.imread(labelh_path, 0)
            line = self.preprocessLine(line)
            line = torch.from_numpy(line).float()
            line_h = self.preprocessLineh(line_h)
            line_h = torch.from_numpy(line_h).float()
        else:
            logging.info("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 5), np.float32)

        sample = {'image': img, 'label': line, 'label_coor': labels,'label_h': line_h}
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample['image'] = self._img_transform(sample['image'])
        # sample['image'] = torch.from_numpy(sample['image']).float()

        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        return sample

    def __len__(self):
        return len(self.img_files)
    def preprocessLine(self, line):
        line = np.asarray(line)
        tmp = np.zeros((1, 80, 80))#for heatmap
        tmp[0, :, :] = line
        line = tmp
        return line
    def preprocessLineh(self, line):
        line = np.asarray(line)
        tmp = np.zeros((1, 160, 160))#for heatmap
        tmp[0, :, :] = line
        line = tmp
        return line
    def _img_transform(self, image):
        image_transforms = ttransforms.Compose([
        ttransforms.ToTensor(),
        ttransforms.Normalize([0.43836477, 0.4470677,  0.42857853], [0.22846407, 0.2287813, 0.24850184]),])
        image = image_transforms(image)
        return image


#  use for test dataloader
if __name__ == "__main__":
    dataloader = torch.utils.data.DataLoader(COCODataset("../data/coco/trainvalno5k.txt",
                                                         (416, 416), True, is_debug=True),
                                             batch_size=2,
                                             shuffle=False, num_workers=1, pin_memory=False)
    for step, sample in enumerate(dataloader):
        for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
            image = image.numpy()
            h, w = image.shape[:2]
            for l in label:
                if l.sum() == 0:
                    continue
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("step{}_{}.jpg".format(step, i), image)
        # only one batch
        break
