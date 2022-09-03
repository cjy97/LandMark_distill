#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2

cv2.setNumThreads(0)
import numpy as np
from logging import getLogger
from torchvision import transforms
import torch.utils.data
from PIL import Image
from PIL import ImageFile
from rich import print

ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

logger = getLogger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_STD = [0.225, 0.224, 0.229]

# Eig vals and vecs of the cov mat
_EIG_VALS = torch.tensor([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = torch.tensor(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )
        return img.add(rgb.view(3, 1, 1).expand_as(img))


class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, data_path, split):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        logger.info("Constructing dataset from {}...".format(split))
        self._data_path, self._split = data_path, split
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        self._imdb, self._class_ids = [], []
        with open(os.path.join(self._data_path, self._split), "r") as fin:
            for line in fin:
                im_dir, cont_id = line.strip().split(" ")
                im_path = os.path.join(self._data_path, im_dir)
                self._imdb.append({"im_path": im_path, "class": int(cont_id)})
                self._class_ids.append(int(cont_id))
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(set(self._class_ids))))

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        # Train and test setups differ
        train_size = 384
        # HWC -> CHW
        im = im.transpose([2, 0, 1])
        if "train" in self._split:
            # Scale and aspect ratio then horizontal flip
            im = transforms.RandomResizedCrop(size=train_size)(im)
            im = transforms.RandomHorizontalFlip(0.5)(im)
        else:
            # Scale and center crop
            im = transforms.Resize(384)(im)
            im = transforms.CenterCrop(train_size)(im)
        # [0, 255] -> [0, 1]
        im = transforms.ToTensor()(im)
        # PCA jitter
        if "train" in self._split:
            im = Lighting(0.1, _EIG_VALS, _EIG_VECS)(im)
        # Color normalization
        im = transforms.Normalize(mean=_MEAN, std=_STD)(im)
        return im

    def __getitem__(self, index):
        # Load the image
        try:
            im = cv2.imread(self._imdb[index]["im_path"])
            im = im.astype(np.float32, copy=False)
        except:
            # print('error: ', self._imdb[index]["im_path"])
            logger.error("error: ", self._imdb[index]["im_path"])
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        # Retrieve the label
        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def default_filelist_reader(filelist, offset=0):

    img_list = []
    with open(filelist, "r", encoding="utf-8") as rf:
        for line in rf.readlines():
            img_path = line.strip().split("\t")
            img_list.append((img_path[0], img_path[2]))  # just for openset test

    return img_list


def train_filelist_reader(filelist, offset=0):

    img_list = []
    with open(filelist, "r", encoding="utf-8") as rf:
        for line in rf.readlines():
            img_path = line.strip().split("\t")[0]
            img_label = int(line.strip().split("\t")[2]) + offset
            img_list.append((img_path, img_label))

    return img_list


def default_filelist_reader_for_multiList(filelist, offset=0, list_idx=0):
    with open(filelist, "r", encoding="utf-8") as file:
        all_lines = file.readlines()
        key_label_list = []
        ids = set()

        for line in all_lines:
            key = line.strip().split("\t")[0]
            label_now = 0  # only for openset
            ids.add(label_now)
            key_label_list.append((list_idx, key, label_now))
    return list(ids), key_label_list


def train_filelist_reader_for_multiList(filelist, offset=0, list_idx=0):
    with open(filelist, "r", encoding="utf-8") as file:
        all_lines = file.readlines()
        key_label_list = []
        ids = set()

        for line in all_lines:
            key, _, label = line.strip().split("\t")
            label_now = int(label) + offset
            ids.add(label_now)
            key_label_list.append((list_idx, key, label_now))
    return list(ids), key_label_list


def test_openset_filelists_reader(filelists, offset_init=0):
    offset = offset_init
    ids = []
    key_label_list = []
    filelist_count = 0
    for filelist in filelists:
        assert os.path.isfile(filelist)
        ids_temp, key_label_list_temp = default_filelist_reader_for_multiList(
            filelist, offset=offset, list_idx=filelist_count
        )
        ids.extend(ids_temp)
        key_label_list.extend(key_label_list_temp)
        offset = len(ids) + offset_init
        filelist_count += 1
        print(filelist)
        print("INFO: ID offset:", offset, " filelist length:", len(key_label_list_temp))
    return ids, key_label_list


def default_filelists_reader(filelists, offset_init=0):
    offset = offset_init
    ids = []
    key_label_list = []
    filelist_count = 0
    for filelist in filelists:
        assert os.path.isfile(filelist)
        ids_temp, key_label_list_temp = train_filelist_reader_for_multiList(
            filelist, offset=offset, list_idx=filelist_count
        )
        ids.extend(ids_temp)
        key_label_list.extend(key_label_list_temp)
        offset = len(ids) + offset_init
        filelist_count += 1
        print(filelist)
        print("INFO: ID offset:", offset, " filelist length:", len(key_label_list_temp))
    return ids, key_label_list


class ImageDataset_MultiList(torch.utils.data.Dataset):
    def __init__(
        self,
        root_paths,
        filelist_paths,
        shuffle=False,
        filelists_reader=default_filelists_reader,
        train=False,
        offset_init=0,
        label2Flabel_path="",
    ):
        self.root_paths = root_paths
        assert len(root_paths) == len(filelist_paths)
        self.ids, self.key_label_list = filelists_reader(filelist_paths, offset_init)
        self.length = len(self.key_label_list)
        self.label_num = int(self.ids[-1] - self.ids[0] + 1)
        self.label_list = [tup[-1] for tup in self.key_label_list]
        self.train = train
        self.offset = offset_init
        if shuffle:
            random.shuffle(self.key_label_list)
        self.label2Flabel_path = label2Flabel_path
        if self.label2Flabel_path:
            self.label2Flabel = torch.load(label2Flabel_path)
            print("loaded label2Flabel:{}".format(label2Flabel_path))

    def _prepare_im(self, im, imgpath):
        """Prepares the image for network input."""
        # Train and test setups differ
        train_size = 384
        # [0, 255] -> [0, 1]

        if self.train:
            # Scale and aspect ratio then horizontal flip
            try:
                im = transforms.RandomResizedCrop(size=(train_size, train_size))(im)
                im = transforms.RandomHorizontalFlip(0.5)(im)
            except Exception as e:
                print(e)
                logger.error("img empty:{}".format(imgpath))

        else:
            # Scale and center crop
            im = transforms.Resize((train_size, train_size))(im)
            im = transforms.CenterCrop((train_size, train_size))(im)
        im = transforms.ToTensor()(im)
        # PCA jitter
        if self.train:
            im = Lighting(0.1, _EIG_VALS, _EIG_VECS)(im)
        im = transforms.Normalize(mean=_MEAN, std=_STD)(im)
        # Color normalization
        return im

    def __getitem__(self, index):
        list_idx, key, id_now = self.key_label_list[index]
        imgpath = os.path.join(self.root_paths[list_idx], key)
        target = int(id_now)
        if self.label2Flabel_path:
            # label2FinalLabel
            if target in self.label2Flabel:
                # print('target {} -> {}'.format(target, self.label2Flabel[target][0] ))
                target = self.label2Flabel[target][0]  # 0为idx 1为分数

        img_pil = Image.fromarray(
            np.array(Image.open(imgpath.encode("UTF-8")).convert("RGB"))[:, :, ::-1]
        )
        # if im is None:
        #     logger.error('img empty:{}'.format(imgpath))
        im = self._prepare_im(img_pil, imgpath)

        return im, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.root_paths + ")"


class ImageDataset(torch.utils.data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root,
        filelist,
        filelist_reader=train_filelist_reader,
        train=False,
        offset=0,
    ):
        # classes, class_to_idx = find_classes(root)
        # imgs = make_dataset(root, class_to_idx)
        # if len(imgs) == 0:
        #     raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
        #                        "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imglist = filelist_reader(filelist, offset=offset)
        self.train = train
        self.label_num = int(self.imglist[-1][-1]) - int(self.imglist[0][-1]) + 1
        self.label_list = [tup[-1] for tup in self.imglist]
        self.offset = offset
        # self.transform = transform
        # self.target_transform = target_transform
        # self.loader = loader
        print(filelist)
        print("INFO: Loaded imglist:", len(self.imglist))
        print("INFO: label_num:", self.label_num)
        # print("INFO: label_num:",self.imglist)

        # print(self.imglist)

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        # Train and test setups differ
        train_size = 384
        # im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]

        if self.train:
            # Scale and aspect ratio then horizontal flip
            try:
                im = transforms.RandomResizedCrop(size=(train_size, train_size))(im)
                im = transforms.RandomHorizontalFlip(0.5)(im)
            except Exception as e:
                print(e)
                logger.error("img empty:{}".format(0))
        else:
            # Scale and center crop
            im = transforms.Resize((train_size, train_size))(im)
            im = transforms.CenterCrop((train_size, train_size))(im)
        # HWC -> CHW
        im = transforms.ToTensor()(im)

        # PCA jitter
        if self.train:
            im = Lighting(0.1, _EIG_VALS, _EIG_VECS)(im)
        im = transforms.Normalize(mean=_MEAN, std=_STD)(im)
        # Color normalization
        return im

    def __getitem__(self, index):

        path = self.imglist[index][0]
        target = int(self.imglist[index][1])
        imgpath = os.path.join(self.root, path)
        img_pil = Image.fromarray(
            np.array(Image.open(imgpath.encode("UTF-8")).convert("RGB"))[:, :, ::-1]
        )
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        im = self._prepare_im(img_pil)
        return im, target

    def __len__(self):
        return len(self.imglist)