from torch.utils import data
import torchvision
import numpy as np
import logging
import random
import math
import cv2
import torchvision.transforms as transforms
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iter=None, crop_size=(321, 321),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        logging.basicConfig(level=logging.DEBUG,
                            filename='dataset.log',
                            filemode='w')
        self.logger = logging.getLogger(__name__)
        self.root = root
        self.list_path = list_path
        self.max_iter = max_iter
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if max_iter is not None:
            self.img_ids = self.img_ids * int(math.ceil(float(max_iter) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            img_file = os.path.join(self.root, image_path)
            label_file = os.path.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.id2train_id = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                            14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                            18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                            28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    @staticmethod
    def generate_scale_label(image, label):
        f_scale = 0.7 * random.randint(0, 14) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def id_to_train_id(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id2train_id.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id2train_id.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        data_files = self.files[index]
        self.logger.info("======> data files: {} ".format(data_files))
        image = cv2.imread(data_files["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(data_files["label"], cv2.IMREAD_GRAYSCALE)
        # self.logger.info("======> image first loaded in \n" + str(image))
        # self.logger.info("======> label first loaded in \n" + str(label))
        label = self.id_to_train_id(label)
        # self.logger.info("======> id to train id \n" + str(label))
        size = image.shape
        self.logger.info("======> size {} ".format(size))
        name = data_files["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        self.logger.info("======> image size after scale: {} ".format(image.shape))
        # self.logger.info("======> image after generate scale and mean \n" + str(image))
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        self.logger.info("======> pad_h: {}, pad_w: {}".format(pad_h, pad_w))
        if pad_h > 0 or pad_w > 0:
            image_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                           cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w,
                                           cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
        else:
            image_pad, label_pad = image, label
        # self.logger.info("======> image after pad \n" + str(image_pad))
        img_h, img_w = label_pad.shape
        self.logger.info("======> img_h: {}, img_w: {}".format(img_h, img_w))
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        self.logger.info("======> h_off: {}, w_off: {}".format(h_off, w_off))
        image = np.asarray(image_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        self.logger.info("======> image size after offset: {} ".format(image.shape))
        # self.logger.info("======> image after offset \n" + str(image))
        image = image.transpose((2, 0, 1))
        self.logger.info("======> image size after transpose: {} ".format(image.shape))
        # self.logger.info("======> image after transpose \n" + str(image))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            self.logger.info("======> flip: {} ".format(flip))
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            self.logger.info("======> image size after flip: {} ".format(image.shape))
            # self.logger.info("======> image after flip \n" + str(image))
        self.logger.info("======> label size: {} \n".format(label.shape))
        self.logger.info("======> image \n" + str(image))
        self.logger.info("======> label \n" + str(label))
        return image.copy(), label.copy(), np.array(size), name


if __name__ == '__main__':
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    dataset = CSDataSet(root='../../', list_path='./list/cityscapes/train.lst', max_iter=40000 * 8,
                        crop_size=(512, 512), mean=IMG_MEAN)
    dataset.__getitem__(0)
    # CIFar10path = '../../'
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_dataset = torchvision.datasets.CIFAR10(root=CIFar10path,
    #                                              train=True,
    #                                              transform=transform,
    #                                              download=False)
    # train_loader = data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    # index = 15
    # print(images[index].numpy())
    # print(labels[index].numpy())
