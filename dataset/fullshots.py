from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, Normalize, ToTensor, ToPILImage, ColorJitter, Grayscale, \
    GaussianBlur
import os
import json
import numpy as np
import torch
from PIL import Image
import h5py
import lmdb


class FullShotsDataset_H5DF(Dataset):
    def __init__(self, dataset_path, stage, label_type, label_suffix='.json', use_data_augment=False, use_resize=False,
                 resize=None,
                 dataset_cache=None, skip_frame=False, skip_flow=False, skip_seg=False, skip_saliency=False,clip_n=16, **kwargs):

        assert stage in ['train', 'test']
        # assert label_type in ['movement', 'scale']
        assert os.path.exists(dataset_path)
        # assert os.output_dir.exists(skip_file_list)
        super(FullShotsDataset_H5DF, self).__init__()
        self.dataset_path = dataset_path
        self.stage = stage
        self.use_data_augment = use_data_augment
        self.use_resize = use_resize
        self.dataset_cache = dataset_cache
        self.skip_frame = skip_frame
        self.skip_flow = skip_flow
        self.skip_seg = skip_seg
        self.skip_saliency = skip_saliency
        self.clip_n=clip_n
        assert self.clip_n in [8,16]
        if not type(label_type) == list:
            self.label_type = [label_type]
        else:
            self.label_type = label_type

        label_dict={'movement':'camera motion','scale':'shot scale','angle':"shot angle",'composition':'shot composition'}
        self.label_type=[label_dict[t] for t in self.label_type]
        self.label_suffix = label_suffix

        if self.use_data_augment:
            self.augment = [ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                            Grayscale(num_output_channels=3), GaussianBlur(kernel_size=3), None]
            self.augment_weight = torch.tensor([5, 1, 1, 3]).float()
            # self.augment_weight = torch.tensor([0, 0, 0, 1]).float()
        if self.use_resize:
            self.resize = Resize(resize)

        self.init()

    def init(self):
        self.total_num = 34251
        self.shape = (3, self.clip_n, 3, 108, 192)
        self.data_dir = os.path.join(self.dataset_path, 'packed')
        self.data = h5py.File(os.path.join(self.data_dir, 'data.hdf5'), mode='r', swmr=True)
        has_cache = False
        if self.dataset_cache is not None:
            if os.path.exists(self.dataset_cache):
                self.using_labels = []
                has_cache = True
        if has_cache:
            print(self.stage, len(self.using_labels))
            return
        with open(os.path.join(os.path.join(self.dataset_path, 'packed', 'label.json')), encoding='utf-8',
                  mode='r') as f:
            self.label_list = json.loads(f.read())
        self.using_labels = []
        for label in self.label_list:
            if len(self.label_type) == 2:
                if self.label_list[label].get('type') == self.stage and self.label_list[label].get(
                        self.label_type[0]) is not None and label.get(
                    self.label_type[1]) is not None:
                    self.using_labels.append(label)
            else:
                if self.label_list[label].get(self.label_type[0]) is not None and self.label_list[label].get(
                        'type') == self.stage:
                    self.using_labels.append(label)
        print(self.stage, len(self.using_labels))

    def __getitem__(self, item):
        sample_name = self.using_labels[item]
        if not self.skip_frame:
            frame = torch.tensor(np.array(self.data[sample_name]['frame']))
        else:
            frame = torch.zeros(1)
        if self.use_data_augment and self.stage == 'train':
            augument = self.augment[torch.multinomial(self.augment_weight, 1).item()]
            if augument is not None and not self.skip_frame:
                frame = torch.stack([augument(img) for img in frame])
        if self.use_resize and not self.skip_frame:
            frame = torch.stack([self.resize(img) for img in frame])

        flow = torch.tensor(np.array(self.data[sample_name]['flow'])) if not self.skip_flow else torch.zeros(1)
        seg = torch.tensor(np.array(self.data[sample_name]['seg'])) if not self.skip_seg else torch.zeros(1)
        saliency = torch.tensor(np.array(self.data[sample_name]['saliency'])) if not self.skip_saliency else torch.zeros(1)
        if self.use_resize:
            flow = torch.stack([self.resize(img) for img in flow]) if not self.skip_flow else flow
            seg = torch.stack([self.resize(img) for img in seg]) if not self.skip_seg else seg
            saliency = torch.stack([self.resize(img) for img in saliency]) if not self.skip_saliency else saliency

        if len(self.label_type) == 1:
            label = self.label_list[sample_name][self.label_type[0]]['value']  # 标签
            if self.clip_n==16:
                return frame, flow, seg, saliency, label  # torch.Size([8, 3, w,h])
            elif self.clip_n == 8:
                return frame[::2,:,:,:], flow[::2,:,:,:], seg[::2,:,:,:], saliency[::2,:,:,:], label  # torch.Size([8, 3, w,h])


        elif len(self.label_type) == 2:
            label1 = self.label_list[sample_name][self.label_type[0]]['value']
            label2 = self.label_list[sample_name][self.label_type[1]]['value']
            if self.clip_n==16:
                return frame, flow, seg, saliency, label1, label2  # torch.Size([8, 3, w,h])
            elif self.clip_n==8:
                return frame[::2,:,:,:], flow[::2,:,:,:], seg[::2,:,:,:], saliency[::2,:,:,:], label1, label2  # torch.Size([8, 3, w,h])



    def __len__(self):
        return len(self.using_labels)


class FullShotsDataset(Dataset):
    def __init__(self, dataset_path, stage, label_type, label_suffix='.json', use_data_augment=False, use_resize=False,
                 resize=None,
                 dataset_cache=None, skip_frame=False, skip_flow=False, skip_seg=False, skip_saliency=False,clip_n=16, **kwargs):

        assert stage in ['train', 'test']
        # assert label_type in ['movement', 'scale']
        assert os.path.exists(dataset_path)
        # assert os.output_dir.exists(skip_file_list)
        super(FullShotsDataset, self).__init__()
        self.dataset_path = dataset_path
        self.stage = stage
        self.use_data_augment = use_data_augment
        self.use_resize = use_resize
        self.dataset_cache = dataset_cache
        self.skip_frame = skip_frame
        self.skip_flow = skip_flow
        self.skip_seg = skip_seg
        self.skip_saliency = skip_saliency
        self.clip_n=clip_n
        assert self.clip_n in [8,16]
        if not type(label_type) == list:
            self.label_type = [label_type]
        else:
            self.label_type = label_type

        label_dict = {'movement': 'camera motion', 'scale': 'shot scale', 'angle': "shot angle",
                      'composition': 'shot composition'}

        self.label_type=[label_dict[t] for t in self.label_type]
        self.label_suffix = label_suffix

        if self.use_data_augment:
            self.augment = [ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                            Grayscale(num_output_channels=3), GaussianBlur(kernel_size=3), None]
            self.augment_weight = torch.tensor([5, 1, 1, 3]).float()
            # self.augment_weight = torch.tensor([0, 0, 0, 1]).float()
        if self.use_resize:
            self.resize = Resize(resize)

        self.init()

    def init(self):
        self.total_num = 34251
        self.shape = (3, self.clip_n, 3, 108, 192)
        self.data_dir = os.path.join(self.dataset_path, 'packed')
        self.data = h5py.File(os.path.join(self.data_dir, 'data.hdf5'), mode='r', swmr=True)
        has_cache = False
        if self.dataset_cache is not None:
            if os.path.exists(self.dataset_cache):
                self.using_labels = []
                has_cache = True
        if has_cache:
            print(self.stage, len(self.using_labels))
            return
        with open(os.path.join(os.path.join(self.dataset_path, 'packed', 'label.json')), encoding='utf-8',
                  mode='r') as f:
            self.label_list = json.loads(f.read())
        self.using_labels = []
        for label in self.label_list:
            if len(self.label_type) == 2:
                if self.label_list[label].get('type') == self.stage and self.label_list[label].get(
                        self.label_type[0]) is not None and label.get(
                    self.label_type[1]) is not None:
                    self.using_labels.append(label)
            else:
                if self.label_list[label].get(self.label_type[0]) is not None and self.label_list[label].get(
                        'type') == self.stage:
                    self.using_labels.append(label)
        print(self.stage, len(self.using_labels))

    def __getitem__(self, item):
        sample_name = self.using_labels[item]
        # print(sample_name)
        if not self.skip_frame:
            frame = torch.tensor(np.array(self.data[sample_name]['frame']))
        else:
            frame = torch.zeros(1)
        if self.use_data_augment and self.stage == 'train':
            augument = self.augment[torch.multinomial(self.augment_weight, 1).item()]
            if augument is not None and not self.skip_frame:
                frame = torch.stack([augument(img) for img in frame])
        if self.use_resize and not self.skip_frame:
            frame = torch.stack([self.resize(img) for img in frame])

        flow = torch.tensor(np.array(self.data[sample_name]['flow'])) if not self.skip_flow else torch.zeros(1)

        if self.use_resize:
            flow = torch.stack([self.resize(img) for img in flow]) if not self.skip_flow else flow


        if len(self.label_type) == 1:
            label = self.label_list[sample_name][self.label_type[0]]['value']
            if self.clip_n==16:
                return frame, flow, label  # torch.Size([8, 3, w,h])
            elif self.clip_n == 8:
                return frame, flow, label  # torch.Size([8, 3, w,h])


        elif len(self.label_type) == 2:
            label1 = self.label_list[sample_name][self.label_type[0]]['value']
            label2 = self.label_list[sample_name][self.label_type[1]]['value']
            if self.clip_n==16:
                return frame, flow, label1, label2  # torch.Size([8, 3, w,h])
            elif self.clip_n==8:
                return frame[::2,:,:,:], flow[::2,:,:,:], label1, label2  # torch.Size([8, 3, w,h])



    def __len__(self):
        return len(self.using_labels)

if __name__ == '__main__':
    d = FullShotsDataset(r'F:\\', stage='test', label_type=['movement'])
