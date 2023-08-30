import torch.utils.data as data
from PIL import Image, ImageFile
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import random
import numpy as np
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""Implementation of LHNet from Shenghai Yuan et al. (ACM MM 2023)."""
class TrainData_train(data.Dataset):
    def __init__(self, dataset_name, crop_size, train_data_dir):
        super().__init__()
        train_dir = train_data_dir + '/' + dataset_name + '/train_' + dataset_name
        train_list = train_dir + '/trainlist.txt'
        with open(train_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.dataset_name = dataset_name
        self.crop_size = crop_size
        self.train_data_dir = train_dir
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        if self.dataset_name == 'NH' or self.dataset_name == 'Dense':
            haze = Image.open(self.train_data_dir + '/haze/' + haze_name)
            clear = Image.open(self.train_data_dir + '/clear_images/' + gt_name + '_GT.png')
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size_w, self.size_h))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        elif self.dataset_name == 'ITS' or self.dataset_name == 'OTS':
            haze = Image.open(self.train_data_dir + '/haze/' + haze_name)
            haze = haze.resize((self.size_w, self.size_h))
            clear = Image.open(self.train_data_dir + '/clear_images/' + gt_name + '.png')
            clear = clear.resize((self.size_w, self.size_h))
        else:
            print('The dataset is not included in this work.')

        haze, gt = self.augData(haze.convert("RGB"), clear.convert("RGB"))

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt

    def augData(self, data, target):
        rand_hor = random.randint(0, 1)
        rand_rot = random.randint(0, 3)
        data = tfs.RandomHorizontalFlip(rand_hor)(data)
        target = tfs.RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            data = FF.rotate(data, 90 * rand_rot)
            target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)


class ValData_train(data.Dataset):
    def __init__(self, dataset_name, crop_size, val_data_dir):
        super().__init__()
        val_dir = val_data_dir + '/' + dataset_name + '/valid_' + dataset_name
        val_list = val_dir + '/val_list.txt'
        self.dataset_name = dataset_name
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            if self.dataset_name == 'NH' or self.dataset_name == 'Dense':
                gt_names = [i.split('_')[0] + '_GT.png' for i in haze_names]  # haze_names#
            elif self.dataset_name == 'OTS' or self.dataset_name == 'ITS':
                gt_names = [i.split('_')[0] + '.png' for i in haze_names]
            else:
                print('The dataset is not included in this work.')
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_dir
        self.data_list = val_list
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + '/input/' + haze_name)
        gt_img = Image.open(self.val_data_dir + '/gt/' + gt_name)
        transform_haze = Compose(
            [ToTensor(), Resize((self.size_w, self.size_h)), Normalize((0.64, 0.6, 0.58), (0.14, 0.15, 0.152))])
        transform_gt = Compose([ToTensor(), Resize((self.size_w, self.size_h))])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

class ValData_test(data.Dataset):
    def __init__(self, dataset_name,val_data_dir):
        super().__init__()
        self.dataset_name = dataset_name
        val_dir = val_data_dir + '/' + dataset_name + '/valid_' + dataset_name + '/'
        val_list = os.path.join(val_dir, 'val_list.txt')
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            if self.dataset_name=='NH' or self.dataset_name=='Dense':
                gt_names = [i.split('_')[0] + '_GT.png' for i in haze_names] #haze_names#
            elif self.dataset_name=='ITS' or self.dataset_name=='OTS':
                gt_names = [i.split('_')[0] + '.png' for i in haze_names]
            else:
                gt_names = None
                print('The dataset is not included in this work.')
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_dir
        self.data_list=val_list

    def get_images(self, index):
        haze_name = self.haze_names[index]

        # build the folder of validation/test data in our way
        if os.path.exists(os.path.join(self.val_data_dir, 'input')):
            haze_img = Image.open(os.path.join(self.val_data_dir, 'input', haze_name))
            if os.path.exists(os.path.join(self.val_data_dir, 'gt')) :
                gt_name = self.gt_names[index]
                gt_img = Image.open(os.path.join(self.val_data_dir, 'gt', gt_name)) ##
                a = haze_img.size
                a_0 =a[1] - np.mod(a[1],16)
                a_1 =a[0] - np.mod(a[0],16)
                haze_crop_img = haze_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = gt_img.crop((0, 0, 0 + a_1, 0+a_0))
                transform_haze = Compose([ToTensor() , Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))])
                transform_gt = Compose([ToTensor()])
                haze_img = transform_haze(haze_crop_img)
                gt_img = transform_gt(gt_crop_img)
            else:
                # the inputs is used to calculate PSNR.
                a = haze_img.size
                a_0 =a[1] - np.mod(a[1],16)
                a_1 =a[0] - np.mod(a[0],16)
                haze_crop_img = haze_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = haze_crop_img
                transform_haze = Compose([ToTensor() , Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))])
                transform_gt = Compose([ToTensor()])
                haze_img = transform_haze(haze_crop_img)
                gt_img = transform_gt(gt_crop_img)
        # Any folder containing validation/test images
        else:
            haze_img = Image.open(os.path.join(self.val_data_dir, haze_name))
            a = haze_img.size
            a_0 =a[1] - np.mod(a[1],16)
            a_1 =a[0] - np.mod(a[0],16)
            haze_crop_img = haze_img.crop((0, 0, 0 + a_1, 0+a_0))
            gt_crop_img = haze_crop_img
            transform_haze = Compose([ToTensor() , Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))])
            transform_gt = Compose([ToTensor()])
            haze_img = transform_haze(haze_crop_img)
            gt_img = transform_gt(gt_crop_img)
        return haze_img, gt_img, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)