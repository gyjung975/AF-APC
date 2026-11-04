import numpy as np
import os
from torch.utils.data import Dataset
import torch
from util.pointnet_utils import normalize_points_np, random_sample_points_np
from util.augmentation import rotate_point_cloud, jitter_point_cloud
import random
import pickle
from collections import defaultdict
from apc import SORDefense
import h5py

ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']


def inference_data(data_root, partition='train'):
    npz = np.load(data_root, allow_pickle=True)
    if partition == 'train':
        return npz['train_pc'], npz['train_label']
    elif partition == 'attack':
        return npz['test_pc'], npz['test_label'], npz['target_label']
    elif partition == 'dis_attack':
        return npz['test_pc'], npz['test_label'], npz['target_label'], npz['ori_pc']
    else:
        return npz['test_pc'], npz['test_label']


def load_data(type, split='test', partition='train', num_point=1024):
    all_data = []
    all_label = []

    if split == 'train':
        split = 'training'
    if type == 'obj_only':
        h5_name = f'data/ScanObjectNN/main_split_nobg/{split}_objectdataset.h5'
    elif type == 'obj_bg':
        h5_name = f'data/ScanObjectNN/main_split/{split}_objectdataset.h5'
    elif type == 'hardest':
        h5_name = f'data/ScanObjectNN/main_split/{split}_objectdataset_augmentedrot_scale75.h5'

    if num_point == 1024:
        h5_name = h5_name.replace('.h5', '_1024.h5')

    print('Load processed data from %s...' % h5_name)
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    if partition == 'attack':
        return all_data, all_label, all_label
    elif partition == 'dis_attack':
        return all_data, all_label, all_label, all_data
    return all_data, all_label


class ScanObjectNN(Dataset):
    """General ScanObjectNN dataset class."""

    def __init__(self, args, sonn_type, num_points, normalize=False,
                 partition='train', augmentation=None, victim=False):
        assert partition in ['train', 'test']
        self.data, self.label = load_data(sonn_type, split=partition)

        self.victim = victim
        if not self.victim:
            datum = defaultdict(list)
            for d, l in zip(self.data, self.label):
                label = l.item()
                datum[label].append(d)

            data, label = [], []
            for i in range(len(datum)):
                d = datum[i]
                nsample = round(len(d) * args.nadv)
                idx = random.sample(range(len(d)), nsample)
                d = [d[j] for j in idx]
                data.extend(d)
                label.extend([i] * len(d))
            self.data, self.label = np.stack(data), np.array(label)
        else:
            print('>>> Sample count not specified, using all data.')

        self.num_points = num_points
        self.normalize = normalize
        self.partition = partition
        self.augmentation = (partition == 'train') if augmentation is None else augmentation

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3] and its label as a scalar."""
        pc = self.data[item][:, :3]
        if self.partition == 'test':
            pc = pc[:self.num_points]
        else:
            pc = random_sample_points_np(pc, self.num_points)

        label = self.label[item]

        # if self.normalize:
        #     pc = normalize_points_np(pc)

        if self.augmentation:
            pc = rotate_point_cloud(pc)
            pc = jitter_point_cloud(pc)

        if not self.victim: return pc, label, pc
        else:               return pc, label

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNNAttack(Dataset):
    """General ScanObjectNN dataset class."""
    def __init__(self, data_root, num_points, sonn_type, split, normalize=False):
        if 'result' in data_root:
            self.data, self.label, self.target = inference_data(data_root, partition='attack')
        else:
            self.data, self.label, self.target = load_data(sonn_type, split=split, partition='attack')
        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        pc = self.data[item][:self.num_points, :3]
        label = self.label[item]
        target = self.target[item]

        # if self.normalize:
        #     pc = normalize_points_np(pc)

        return pc, label, target

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNNAPC(Dataset):
    """General ScanObjectNN dataset class."""
    def __init__(self, args, num_points, attack, partition, normalize=False):
        assert partition in ['train', 'test']
        adv_root = f'result/{args.sonn_type}/{attack.lower()}_{args.model}/{attack.lower()}_{partition}.npz'
        print(f'Load data from {adv_root}...')
        self.data, self.label, self.target, self.orig_pc = inference_data(adv_root, partition='dis_attack')
        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3] and its label as a scalar."""
        pc = self.data[item]
        pc = random_sample_points_np(pc, self.num_points)
        label = self.label[item]

        orig_pc = self.orig_pc[item][:self.num_points, :3]

        # if self.normalize:
        #     pc = normalize_points_np(pc)

        return pc, label, orig_pc

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNNAPCHybrid(Dataset):
    """General ScanObjectNN dataset class."""
    def __init__(self, args, num_points, attack, partition, normalize=False):
        assert partition in ['train', 'test']
        adv_root = f'result/{args.sonn_type}/{attack.lower()}_{args.model}/{attack.lower()}_{partition}.npz'
        print(f'Load data from {adv_root}...')
        self.data, self.label, self.target, self.orig_pc = inference_data(adv_root, partition='dis_attack')

        adv_data = defaultdict(list)
        ori_data = defaultdict(list)
        for adv, l, ori in zip(self.data, self.label, self.orig_pc):
            label = l.item()
            adv_data[label].append(adv)
            ori_data[label].append(ori)

        data, label, orig_pc = [], [], []
        for i in range(len(ori_data)):
            ori, adv = ori_data[i], adv_data[i]
            assert len(ori) == len(adv)
            nsample = round(len(ori) * args.nadv)
            idx = random.sample(range(len(ori)), nsample)
            ori, adv = [ori[j] for j in idx], [adv[j] for j in idx]
            data.extend(adv)
            label.extend([i] * len(adv))
            orig_pc.extend(ori)
        self.data, self.label, self.orig_pc = np.stack(data), np.array(label), np.stack(orig_pc)

        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3] and its label as a scalar."""
        pc = self.data[item]
        pc = random_sample_points_np(pc, self.num_points)
        label = self.label[item]

        orig_pc = self.orig_pc[item][:self.num_points, :3]
        # if self.normalize:
        #     pc = normalize_points_np(pc)

        return pc, label, orig_pc

    def __len__(self):
        return self.data.shape[0]
