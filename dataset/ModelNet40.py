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


def load_data(data_root, partition='train'):
    print('Load processed data from %s...' % data_root)
    with open(data_root, 'rb') as f:
        list_of_points, list_of_labels = pickle.load(f)
        data, label = np.stack(list_of_points), np.concatenate(list_of_labels)
    if partition == 'attack':
        return data, label, label
    elif partition == 'dis_attack':
        return data, label, label, data
    else:
        return data, label


class ModelNet40(Dataset):
    """General ModelNet40 dataset class."""
    def __init__(self, args, data_root, num_points, normalize=True,
                 partition='train', augmentation=None, victim=False):
        assert partition in ['train', 'test']
        self.data, self.label = load_data(data_root, partition=partition)

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

        if self.normalize:
            pc = normalize_points_np(pc)

        if self.augmentation:
            pc = rotate_point_cloud(pc)
            pc = jitter_point_cloud(pc)

        if not self.victim: return pc, label, pc
        else:               return pc, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40Attack(Dataset):
    """Modelnet40 dataset for target attack evaluation.
    We return an additional target label for an example.
    """
    def __init__(self, data_root, num_points, normalize=True):
        if 'result' in data_root:
            self.data, self.label, self.target = inference_data(data_root, partition='attack')
        else:
            self.data, self.label, self.target = load_data(data_root, partition='attack')
        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        pc = self.data[item][:self.num_points, :3]
        label = self.label[item]
        target = self.target[item]

        if self.normalize:
            pc = normalize_points_np(pc)

        return pc, label, target

    def __len__(self):
        return self.data.shape[0]


class ModelNet40NormalAttack(Dataset):
    """Modelnet40 dataset with point normals and target label."""
    def __init__(self, data_root, num_points, normalize=True):
        self.data, self.label, self.target = load_data(data_root, partition='attack')
        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 6], its label as a scalar
            and its target label for attack as a scalar.
        """
        pc = self.data[item][:self.num_points, :6]
        label = self.label[item]
        target = self.target[item]

        if self.normalize:
            pc[:, :3] = normalize_points_np(pc[:, :3])

        return pc, label, target

    def __len__(self):
        return self.data.shape[0]


class ModelNet40APC(Dataset):
    """General ModelNet40 dataset class."""
    def __init__(self, args, num_points, attack, partition, normalize=True):
        assert partition in ['train', 'test']
        adv_root = f'result/{args.dataset}/{attack.lower()}_{args.model}/{attack.lower()}_{partition}.npz'
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

        if self.normalize:
            pc = normalize_points_np(pc)

        return pc, label, orig_pc

    def __len__(self):
        return self.data.shape[0]


class ModelNet40APCHybrid(Dataset):
    """General ModelNet40 dataset class."""
    def __init__(self, args, num_points, attack, partition, normalize=True):
        assert partition in ['train', 'test']
        adv_root = f'result/{args.dataset}/{attack.lower()}_{args.model}/{attack.lower()}_{partition}.npz'
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
        if self.normalize:
            pc = normalize_points_np(pc)

        return pc, label, orig_pc

    def __len__(self):
        return self.data.shape[0]


class ModelNet40_Geo():
    def __init__(self, data_mat_file, attack_label='All', resample_num=-1,
                 is_half_forward=False, normalize=True):
        self.normalize = normalize
        self.data_root = data_mat_file
        self.attack_label = attack_label
        self.is_half_forward = is_half_forward

        if not os.path.isfile(self.data_root):
            assert False, 'No exists .npz file!'

        dataset = load_data(self.data_root, partition='attack')
        # data = torch.FloatTensor(dataset['data'])
        # normal = torch.FloatTensor(dataset['normal'])
        # label = dataset['label']
        data = dataset[0][:, :, :3]
        normal = dataset[0][:, :, 3:]
        label = dataset[1][:, np.newaxis]

        if resample_num > 0:
            tmp_data_set = []
            tmp_normal_set = []
            for j in range(data.size(0)):
                tmp_data, tmp_normal = self.__farthest_points_normalized(data[j].t(), resample_num, normal[j].t())
                tmp_data_set.append(torch.from_numpy(tmp_data).t().float())
                tmp_normal_set.append(torch.from_numpy(tmp_normal).t().float())
            data = torch.stack(tmp_data_set)
            normal = torch.stack(tmp_normal_set)

        if attack_label in ten_label_names:
            for k, label_name in enumerate(ten_label_names):
                if attack_label == label_name:
                    self.start_index = k * 25
                    self.data = data[k * 25:(k + 1) * 25]
                    self.normal = normal[k * 25:(k + 1) * 25]
                    self.label = label[k * 25:(k + 1) * 25]
        elif attack_label == 'All':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        # elif attack_label == 'Untarget' or attack_label == 'Random':
        elif attack_label == 'Untarget' or attack_label == 'Random':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        else:
            assert False

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # if (self.attack_label in ten_label_names) or (self.attack_label == 'All'):
        if (self.attack_label in ten_label_names) or (self.attack_label == 'All'):
            label = self.label[index]

            target_labels = []
            for i in ten_label_indexes:
                if label != i:
                    target_labels.append(i)
                else:
                    target_labels.append(i + 1)
            target_labels = torch.IntTensor(np.array(target_labels)).long()
            gt_labels = torch.IntTensor(label).long().expand_as(target_labels)
            assert target_labels.size(0) == 9

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(9, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(9, -1, -1)
            if self.is_half_forward:
                return [[pcs[:4, :, :], normals[:4, :, :], gt_labels[:4], target_labels[:4]],
                        [pcs[4:, :, :], normals[4:, :, :], gt_labels[4:], target_labels[4:]]]
            else:
                return [pcs, normals, gt_labels, target_labels]
        elif self.attack_label == 'Untarget':
            label = self.label[index]
            gt_labels = torch.IntTensor(label).long()

            pc = self.data[index]

            if self.normalize:
                pc = normalize_points_np(pc)
            pc = torch.FloatTensor(pc).contiguous().t()

            pcs = pc.unsqueeze(0).expand(1, -1, -1)

            normal = self.normal[index]
            normal = torch.FloatTensor(normal).contiguous().t()
            normals = normal.unsqueeze(0).expand(1, -1, -1)
            return [pcs, normals, gt_labels]
        elif self.attack_label == 'Random':
            label = self.label[index]
            gt_labels = torch.IntTensor(label).long()

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(1, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(1, -1, -1)

            target_labels = torch.IntTensor([choice([i for i in range(0, 40) if i not in [gt_labels.item()]])]).long()

            return [pcs, normals, gt_labels, target_labels]

    def __farthest_points_normalized(self, obj_points, num_points, normal):
        first = np.random.randint(len(obj_points))
        selected = [first]
        dists = np.full(shape=len(obj_points), fill_value=np.inf)

        for _ in range(num_points - 1):
            dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis=1))
            selected.append(np.argmax(dists))
        res_points = np.array(obj_points[selected])
        res_normal = np.array(normal[selected])

        # normalize the points and faces
        avg = np.average(res_points, axis=0)
        res_points = res_points - avg[np.newaxis, :]
        dists = np.max(np.linalg.norm(res_points, axis=1), axis=0)
        res_points = res_points / dists

        return res_points, res_normal
