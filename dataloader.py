import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
import collections
import json


class ObjDictionary(object):
    def __init__(self, path):
        self.discribe2index = {}
        self.index2discribe = {}
        self.n_chars = 24  # Count

        with open(path, 'r') as json_file:
            data = json.load(json_file)
            for p in data:
                self.discribe2index[p] = int(data[p])
                self.index2discribe[int(data[p])] = p
            json_file.close()

    def get_index(self, discribe):
        return self.discribe2index[discribe]

    def get_discribe(self, index):
        return self.index2discribe[index]

obj_dict = ObjDictionary('objects.json')

def getData(mode):
    if mode == 'train':

        image_name = []
        label = []
        with open('train.json') as json_file:
            data = json.load(json_file)
            for p in data:
                image_name.append(p)
                one_hot_label = np.zeros(obj_dict.n_chars, dtype=np.float32)
                for q in data[p]:
                    index = obj_dict.get_index(q)
                    one_hot_label[index] = 1
                label.append(one_hot_label)

        print('train:')
        print(len(label))
        return np.squeeze(image_name), np.squeeze(label)
    else:
        image_name = []
        label = []
        with open('test.json') as json_file:
            data = json.load(json_file)
            for p in data:
                one_hot_label = np.zeros(obj_dict.n_chars, dtype=np.float32)
                for q in p:
                    index = obj_dict.get_index(q)
                    one_hot_label[index] = 1.0
                label.append(one_hot_label)

        print('test:')
        print(len(label))
        return image_name, np.squeeze(label)


class iclevrDataset(data.Dataset):
    def __init__(self, root, mode, transform):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)
            transform : any transform, e.g., random flipping, rotation, cropping, and normalization.

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.label)

    def __getitem__(self, index):
        """retuen a pair of (image, label) which image is changed from self.tranform """

        if self.mode == 'train':
            img_path = self.root + self.img_name[index]
            img = Image.open(img_path).convert('RGB')
            # img = img.resize( (width, height), Image.BILINEAR )
            label = self.label[index]

            if self.transform is not None:
                img = self.transform(img)

            return img, label
        elif self.mode == 'test':

            label = self.label[index]

            return label
