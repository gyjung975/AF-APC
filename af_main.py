import argparse
import os
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import ModelNet40, ModelNet40APC, ModelNet40APCHybrid, ScanObjectNN, ScanObjectNNAPC, ScanObjectNNAPCHybrid
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointTransformer

from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed

from apc import APC, APCT2R
from attack_scripts.attack import ChamferDist, HausdorffDist, L2Dist
from util.utils import create_logger, accuracy, AverageMeter, ProgressMeter, cosine_lr


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in training')
    parser.add_argument('--num_points', type=int, default=1024)

    parser.add_argument('--dataset', type=str, default='mn40', choices=['mn40', 'sonn'])
    parser.add_argument('--sonn_type', type=str, default='obj_only', choices=['obj_only', 'obj_bg', 'hardest'])
    parser.add_argument('--normalize_pc', action='store_true', default=False)

    parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'pointnet2', 'dgcnn', 'pt2'])
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--feature_transform', action='store_false', default=True)

    parser.add_argument('--trans_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--num_group', type=int, default=64)
    parser.add_argument('--encoder_dims', type=int, default=256)

    parser.add_argument('--attack_type', nargs='+', type=str, default=['synth'])
    parser.add_argument('--attack_type_eval', type=str, default='pgd')

    parser.add_argument('--sor', action='store_true', default=False, help='use SOR for APC')
    parser.add_argument('--sor_k', type=int, default=2)
    parser.add_argument('--sor_alpha', type=float, default=1.1)
    parser.add_argument('--feat_dims', type=int, default=256)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--off_dims', type=int, default=128)
    parser.add_argument('--leaky_relu', type=int, default=1, choices=[0, 1])

    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--feat_loss_weight', type=float, default=5.)
    parser.add_argument('--dist_func', type=str, default='cham', choices=['cham', 'haus'])
    parser.add_argument('--dist_loss_weight', type=float, default=5.)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--lr_start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr_end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)

    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    parser.add_argument('--ncln', type=float, default=0.3)
    parser.add_argument('--nadv', type=float, default=0.3)

    parser.add_argument('--synth_ops', nargs='+', type=str,
                        default=['jitter', 'sparse', 'shift', 'patchdrop'])  # patchdrop == old patchdrop@2 since 2026-10-16
    parser.add_argument('--synth_nops', type=str, default='1,1,1', help='pool to sample #ops per example from')
    parser.add_argument('--synth_severity', type=str, default='1.0', help='global corruption magnitude multiplier')
    parser.add_argument('--synth_frac', type=float, default=-1., help='class-frac of train set used for synth pairs; default 3*nadv')
    return parser.parse_args()


def train(train_loader, model, prompter, criterion, dist_func, optimizer, epoch, lr_scheduler, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    dist_losses = AverageMeter('Dist Loss', ':.4e')
    feat_losses = AverageMeter('Feat Loss', ':.4e')
    def_top1 = AverageMeter('Def Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, dist_losses, feat_losses, def_top1],
        prefix="Epoch: [{}]".format(epoch))

    model.eval()
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    end = time.time()
    for i, (adv_data, label, ori_data) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)

        step = num_batches_per_epoch * epoch + i
        lr_scheduler(step)

        adv_data, label = adv_data.float().cuda(), label.long().cuda()  # (b, n, 6), (b,)
        ori_data = ori_data.float().cuda()

        def_data = prompter(adv_data)        # (b, n, 3)

        ori_data = ori_data.transpose(1, 2).contiguous()
        def_data = def_data.transpose(1, 2).contiguous()        # (b, 3, n)

        def_logits, _, _, def_feat = model(def_data)

        loss = criterion(def_logits, label)

        with torch.no_grad():
            ori_feat = model(ori_data)[-1]
        feat_loss = torch.nn.functional.mse_loss(def_feat, ori_feat)
        dist_loss = dist_func(def_data.transpose(1, 2), ori_data.transpose(1, 2))

        loss = loss + args.feat_loss_weight * feat_loss + args.dist_loss_weight * dist_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), def_data.size(0))
        dist_losses.update(dist_loss.item(), def_data.size(0))
        feat_losses.update(feat_loss.item(), def_data.size(0))

        def_acc1 = accuracy(def_logits, label, topk=(1,))
        def_top1.update(def_acc1[0].item(), def_data.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    logger.info(f'Overall def accuracy: {def_top1.avg}')

    return {'train_def_acc1': def_top1.avg,
            'train_loss': losses.avg, 'train_feat_loss': feat_losses.avg, 'train_dist_loss': dist_losses.avg,
            'lr': optimizer.param_groups[0]['lr']}


def evaluate(test_loader, model, prompter, criterion, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    def_top1 = AverageMeter('Def Acc@1', ':6.2f')

    model.eval()
    prompter.eval()

    end = time.time()
    with torch.no_grad():
        for i, (adv_data, label, ori_data) in enumerate(tqdm(test_loader)):
            adv_data, label = adv_data.float().cuda(), label.long().cuda()  # (b, n, 6), (b,)

            def_data = prompter(adv_data)        # (b, n, 3)

            def_data = def_data.transpose(1, 2).contiguous()        # (b, 3, n)

            def_logits = model(def_data)[0]

            loss = criterion(def_logits, label)
            losses.update(loss.item(), def_data.size(0))

            def_acc1 = accuracy(def_logits, label, topk=(1,))
            def_top1.update(def_acc1[0].item(), def_data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(f'Overall def accuracy: {def_top1.avg}')

        return {'val_loss': losses.avg, 'val_def_acc1': def_top1.avg}


def main():
    args = parse_args()
    set_seed(args.seed)

    ncls = 40 if args.dataset == 'mn40' else 15
    if args.model.lower() == 'dgcnn':
        model = DGCNN(output_channels=ncls, emb_dims=args.emb_dims, k=args.k)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=ncls, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=ncls)
    elif args.model.lower() == 'pt2':
        model = PointTransformer(args, num_classes=ncls)
        args.feat_dims = 128
    else:
        print('Model not recognized')
        exit(-1)

    from config import BEST_WEIGHTS
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_points]
    if args.dataset == 'sonn':
        BEST_WEIGHTS[args.model] = BEST_WEIGHTS[args.model].replace(f"{args.model}/", f"{args.model}_{args.sonn_type}/")
    state_dict = torch.load(BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    adv_trainset, adv_testset = [], []
    if args.dataset == 'mn40':
        ori_data = "data/modelnet40_normal_resampled/modelnet40_{}_1024pts.dat"
        orig_trainset = ModelNet40(args, ori_data.format('train'), num_points=args.num_points,
                                   normalize=True, partition='train', victim=False)
    elif args.dataset == 'sonn':
        orig_trainset = ScanObjectNN(args, sonn_type=args.sonn_type, num_points=args.num_points,
                                     normalize=False, partition='train', victim=False)
    adv_trainset.append(orig_trainset)

    from util.synth_corrupt import SynthCorruptSet
    sfrac = args.synth_frac if args.synth_frac > 0 else 3 * args.nadv
    train_dataset = SynthCorruptSet(args, args.sonn_type, args.num_points,
                                    partition='train', frac=sfrac, dataset=args.dataset)
    test_dataset = SynthCorruptSet(args, args.sonn_type, args.num_points,
                                   partition='test', frac=1.0, dataset=args.dataset)
    adv_trainset.append(train_dataset)
    adv_testset.append(test_dataset)

    train_dataset = torch.utils.data.ConcatDataset(adv_trainset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                              shuffle=True, num_workers=4)
    test_dataset = torch.utils.data.ConcatDataset(adv_testset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True,
                             shuffle=False, num_workers=0)

    hat = ','.join(args.attack_type)
    exp_name = f"{hat}_{args.model}"

    if args.wandb:
        wandb.init(project='APC', name=exp_name)
        wandb.config.update(args)

    logger = create_logger(f'./log/{args.dataset}/{exp_name}', exp_name, 'info') if args.dataset == 'mn40' \
        else create_logger(f'./log/{args.sonn_type}/{exp_name}', exp_name, 'info')
    logger.info(f'args: {args}')
    logger.info(f'\nattack_type: {args.attack_type}')

    model = model.cuda()
    for p in model.parameters():
        p.requires_grad = False

    if args.model == 'pt2':
        args.k = args.group_size
        prompter = APCT2R(args.k, args.feat_dims, args.off_dims, args).cuda()
    else:
        prompter = APC(args.k, args.feat_dims, args.off_dims, args).cuda()

    logger.info(f'#params (victim): {sum(p.numel() for p in model.parameters())}')
    logger.info(f'#params (APC): {sum(p.numel() for p in prompter.parameters())}')

    criterion = nn.CrossEntropyLoss()
    if args.dist_func == 'cham': dist_func = ChamferDist(method='adv2ori')
    else:                        dist_func = HausdorffDist(method='adv2ori')

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(prompter.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(prompter.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(prompter.parameters(), lr=args.lr, betas=args.betas,
                                      eps=args.eps, weight_decay=args.wd)

    total_steps = len(train_loader) * args.epochs
    lr_scheduler = cosine_lr(optimizer, args.lr, args.warmup_epochs, total_steps)

    best_acc1 = 0
    best_epoch = -1
    for epoch in range(args.epochs):
        logger.info(f'\n========== Epoch [{epoch + 1}/{args.epochs}] ==========')
        train_stats = train(train_loader, model, prompter, criterion, dist_func, optimizer, epoch, lr_scheduler, args, logger)

        logger.info('>>> Evaluating... <<<')
        val_stats = evaluate(test_loader, model, prompter, criterion, args, logger)
        acc1 = val_stats['val_def_acc1']

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs or is_best:
            e = 'last' if epoch + 1 == args.epochs else epoch + 1
            saved_data = {
                'epoch': e,
                'state_dict': prompter.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc1': best_acc1
            }
            prefix = f'log/{args.dataset}/{exp_name}' if args.dataset == 'mn40' else f'log/{args.sonn_type}/{exp_name}'
            if is_best:
                print("=> saving best checkpoint")
                torch.save(saved_data, f'{prefix}/checkpoint_best.pt')
                best_epoch = epoch + 1
            else:
                print(f"=> saving {e} checkpoint")
                torch.save(saved_data, f'{prefix}/checkpoint_{e}.pt')

        log_stats = {**{f'{k}': v for k, v in train_stats.items()},
                     **{f'{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_acc1': best_acc1,
                     'best_epoch': best_epoch}

        if args.wandb:
            wandb.log(log_stats)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
