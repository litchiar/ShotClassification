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


#

class MovieShotsDataset(Dataset):
    def __init__(self, dataset_path, stage, label_type, label_suffix='.json', downsample_factor=4,
                 ignore_feature=list()):

        assert stage in ['train', 'test']
        assert label_type in ['movement', 'scale']
        assert os.path.exists(dataset_path)
        super(MovieShotsDataset, self).__init__()
        self.dataset_path = dataset_path
        self.stage = stage
        self.label_type = label_type
        self.label_suffix = label_suffix
        self.ignore_feature = ignore_feature
        self.init()
        downsample_size = (1920 // downsample_factor, 1080 // downsample_factor)
        self.transpose = Compose(
            [ToTensor(), Resize(downsample_size), Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])

    def init(self):
        self.data_list = []
        self.label_list = []
        data_list_path = os.path.join(os.getcwd(), '../cache', f'{self.label_type}_{self.stage}_data_list.json')
        label_list_path = os.path.join(os.getcwd(), '../cache', f'{self.label_type}_{self.stage}_label_list.json')
        if os.path.exists(data_list_path) and os.path.exists(label_list_path):
            with open(data_list_path, encoding='utf-8', mode='r') as data_f:
                self.data_list = [d for d in json.loads(data_f.read())]
            with open(label_list_path, encoding='utf-8', mode='r') as label_f:
                self.label_list = [d for d in json.loads(label_f.read())]
            return
        for tr in ['trailer', 'trailer_v2']:
            for video in os.listdir(os.path.join(self.dataset_path, tr)):
                for shot in os.listdir(os.path.join(self.dataset_path, tr, video)):
                    sample_dir = os.path.join(self.dataset_path, tr, video, shot)
                    # if len(os.listdir(sample_dir)) <= 25:
                    #     continue
                    json_file = os.path.join(self.dataset_path, tr, video, shot, shot + self.label_suffix)
                    if os.path.exists(json_file):
                        with open(json_file, encoding='utf-8', mode='r') as json_f:
                            label = json.loads(json_f.read())
                            if label[self.label_type] is not None and self.stage == label['type']:
                                self.data_list.append(sample_dir)
                                self.label_list.append(label[self.label_type])

        assert len(self.data_list) == len(self.label_list)
        with open(data_list_path, encoding='utf-8', mode='w') as data_f:
            data_f.write(json.dumps(self.data_list))
        with open(label_list_path, encoding='utf-8', mode='w') as label_f:
            label_f.write(json.dumps(self.label_list))

    def merge_label(self):
        if self.label_type == 'movement':
            for t_i, t_label in enumerate(self.label_list):
                if t_label['label'] == 'Multi_movement':
                    self.label_list[t_i] = {'label': 'Motion', 'value': 0}

    def __getitem__(self, item):

        label = self.label_list[item]['value']
        frame = torch.stack(
            [self.transpose(Image.open(os.path.join(self.data_list[item], f'image_{i}.jpg'))) for i in range(8)])

        flow = torch.stack(
            [self.transpose(Image.open(os.path.join(self.data_list[item], f'flow_{i}.jpg'))) for i in range(8)])

        seg = torch.stack(
            [self.transpose(Image.open(os.path.join(self.data_list[item], f'seg_{i}.jpg'))) for i in range(8)])

        return frame, flow, seg, label  # torch.Size([8, 3, h,w])

    def __len__(self):
        return len(self.data_list)



class MovieShotsDataset_H5DF(Dataset):
    def __init__(self, dataset_path, stage, label_type, label_suffix='.json', use_data_augment=False, use_resize=False,
                 resize=None,
                 dataset_cache=None, skip_frame=False, skip_flow=False, skip_seg=False, skip_saliency=False, **kwargs):

        assert stage in ['train', 'test']
        # assert label_type in ['movement', 'scale']
        assert os.path.exists(dataset_path)
        # assert os.output_dir.exists(skip_file_list)
        super(MovieShotsDataset_H5DF, self).__init__()
        self.dataset_path = dataset_path
        self.stage = stage
        self.use_data_augment = use_data_augment
        self.use_resize = use_resize
        self.dataset_cache = dataset_cache
        self.skip_frame = skip_frame
        self.skip_flow = skip_flow
        self.skip_seg = skip_seg
        self.skip_saliency = skip_saliency
        if not type(label_type) == list:
            self.label_type = [label_type]
        else:
            self.label_type = label_type
        self.label_suffix = label_suffix

        if self.use_data_augment:
            self.augment = [ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                            Grayscale(num_output_channels=3), GaussianBlur(kernel_size=3), None]
            self.augment_weight = torch.tensor([5, 1, 1, 3]).float()
        if self.use_resize:
            self.resize = Resize(resize)

        self.init()

    def init(self):
        self.total_num = 34251
        self.shape = (3, 8, 3, 108, 192)
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
        seg = torch.tensor(np.array(self.data[sample_name]['seg'])) if not self.skip_seg else torch.zeros(1)
        saliency = torch.tensor(np.array(self.data[sample_name]['saliency'])) if not self.skip_saliency else torch.zeros(1)
        if self.use_resize:
            flow = torch.stack([self.resize(img) for img in flow]) if not self.skip_flow else flow
            seg = torch.stack([self.resize(img) for img in seg]) if not self.skip_seg else seg
            saliency = torch.stack([self.resize(img) for img in saliency]) if not self.skip_saliency else saliency

        if len(self.label_type) == 1:
            label = self.label_list[sample_name][self.label_type[0]]['value']  # 标签

            return frame, flow, seg, saliency, label  # torch.Size([8, 3, w,h])
        elif len(self.label_type) == 2:
            label1 = self.label_list[sample_name][self.label_type[0]]['value']
            label2 = self.label_list[sample_name][self.label_type[1]]['value']
            return frame, flow, seg, saliency, label1, label2  # torch.Size([8, 3, w,h])

    def __len__(self):
        return len(self.using_labels)



if __name__ == '__main__':
    d = MovieShotsDataset_H5DF(r'F:\\', stage='train', label_type=['scale'])
    from torch.utils.data import DataLoader
    frame,flow,seg,saliency,label=d[0]
    to_pil = ToPILImage()
    dd = DataLoader(d, batch_size=32, shuffle=False)
    from tqdm import tqdm
    labels = []
    # for frame,flow,, label1 in tqdm(dd):
    #     print(frame.shape)
    #
    #
    #     break
    #
    # label = []
    for frame, flow, seg, saliency, label1 in tqdm(dd):

        frame_data = frame[3]
        flow_data = flow[3]
        seg_data = seg[3]
        saliency_data = saliency[3]
        saliency_data=saliency_data.repeat_interleave(3,dim=1)
        print(frame_data.shape)
        print(flow_data.shape)
        print(seg_data.shape)
        print(saliency_data.shape)

        #
        if not os.path.exists(os.path.join(os.getcwd(), 'temp')):
            os.makedirs(os.path.join(os.getcwd(), 'temp'))
        path = os.path.join(os.getcwd(), 'temp')
        for i, frame in enumerate(frame_data):
            to_pil(frame).save(os.path.join(path, f'image_{i}.jpg'))
        for i, frame in enumerate(flow_data):
            to_pil(frame).save(os.path.join(path, f'flow_{i}.jpg'))
        for i, frame in enumerate(seg_data):
            to_pil(frame).save(os.path.join(path, f'seg_{i}.jpg'))
        for i, frame in enumerate(saliency_data):
            to_pil(frame).save(os.path.join(path, f'saliency_{i}.jpg'))
        print(path)
        break
    # print(label)
