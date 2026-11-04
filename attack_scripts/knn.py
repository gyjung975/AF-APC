import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from config import BEST_WEIGHTS
from dataset import ModelNet40NormalAttack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointTransformer

from util.utils import str2bool, set_seed

from attack import CWKNN
from attack import CrossEntropyAdvLoss, LogitsAdvLoss, UntargetedLogitsAdvLoss
from attack import ChamferkNNDist
from attack import ProjectInnerClipLinf


def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    all_ori_pc = []
    num = 0
    for pc, label, target in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), label.long().cuda(non_blocking=True)
            target_label = target.long().cuda(non_blocking=True)

        best_pc, success_num = attacker.attack(pc, label, target_label)

        num += success_num
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())
        all_target_lbl.append(target_label.detach().cpu().numpy())
        all_ori_pc.append(pc.detach().cpu().numpy())

    all_adv_pc = np.concatenate(all_adv_pc, axis=0)
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)
    all_target_lbl = np.concatenate(all_target_lbl, axis=0)
    all_ori_pc = np.concatenate(all_ori_pc, axis=0)
    return all_adv_pc, all_real_lbl, all_target_lbl, num, all_ori_pc


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/data/attack_data.npz')
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pt2'])
    parser.add_argument('--feature_transform', action='store_false', default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--trans_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--num_group', type=int, default=64)
    parser.add_argument('--encoder_dims', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=15.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr', type=float, default=1e-4,
                        help='lr in CW optimization')
    parser.add_argument('--num_iter', type=int, default=2500, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_points]
    set_seed(1)
    print(args)

    cudnn.benchmark = True

    ncls = 40 if args.dataset == 'mn40' else 15
    if args.model.lower() == 'dgcnn':
        model = DGCNN(output_channels=ncls, emb_dims=args.emb_dims, k=args.k)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=ncls, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=ncls)
    elif args.model.lower() == 'pt2':
        model = PointTransformer(args, num_classes=ncls)
    else:
        print('Model not recognized')
        exit(-1)

    state_dict = torch.load(BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    if args.adv_func == 'logits':
        adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()

    dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                               knn_k=5, knn_alpha=1.05,
                               chamfer_weight=5., knn_weight=3.)
    clip_func = ProjectInnerClipLinf(budget=0.1)
    attacker = CWKNN(model, adv_func, dist_func, clip_func,
                     attack_lr=args.attack_lr,
                     num_iter=args.num_iter)

    data_root = f"data/modelnet40_normal_resampled/modelnet40_{args.split}_1024pts.dat"
    test_set = ModelNet40NormalAttack(data_root, num_points=args.num_points, normalize=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)

    attacked_data, real_label, target_label, success_num, all_ori_pc = attack()

    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    save_path = f'result/{args.dataset}/knn_{args.model}'
    os.makedirs(save_path, exist_ok=True)
    save_name = f'knn_{args.split}'
    np.savez(os.path.join(save_path, save_name),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8),
             target_label=target_label.astype(np.uint8),
             ori_pc=all_ori_pc.astype(np.float32))
