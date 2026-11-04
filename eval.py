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

from dataset import ModelNet40APC, ScanObjectNNAPC
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointTransformer

from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed

from apc import APC, APCT2R
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

    parser.add_argument('--apc_model', type=str, default='pointnet', choices=['pointnet', 'pointnet2', 'dgcnn', 'pt2'])
    parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'pointnet2', 'dgcnn', 'pt2'])
    parser.add_argument('--cross_model', action='store_true', default=False)
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--feature_transform', action='store_false', default=True)

    parser.add_argument('--trans_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--num_group', type=int, default=64)
    parser.add_argument('--encoder_dims', type=int, default=256)

    parser.add_argument('--attack_type', nargs='+', type=str, default=['pgd', 'hit'])
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
    return parser.parse_args()


@ torch.no_grad()
def evaluate(test_loader, model, prompter, eval_attack, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    def_top1 = AverageMeter('Def Acc@1', ':6.2f')

    model.eval()
    prompter.eval()

    end = time.time()
    with torch.no_grad():
        for i, (adv_data, label, ori_data) in enumerate(tqdm(test_loader, desc=eval_attack)):
            adv_data, label = adv_data.float().cuda(), label.long().cuda()  # (b, n, 6), (b,)
            def_data = prompter(adv_data)        # (b, n, 3)
            def_data = def_data.transpose(1, 2).contiguous()        # (b, 3, n)
            def_logits = model(def_data)[0]

            def_acc1 = accuracy(def_logits, label, topk=(1,))
            def_top1.update(def_acc1[0].item(), def_data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(f'Overall def accuracy: {def_top1.avg}')
        return {'val_def_acc1': def_top1.avg}


def main():
    args = parse_args()

    if args.attack_type_eval == 'all':
        eval_attacks = ['add', 'cluster', 'perturb', 'knn', 'ifgm', 'pgd', 'hit', 'drop', 'advpc', 'sia', 'geo'] if args.dataset == 'mn40' \
            else ['add', 'cluster', 'perturb', 'ifgm', 'pgd', 'drop', 'advpc']
    else:
        eval_attacks = [args.attack_type_eval]

    cln_acc, def_acc = [], []
    for eval_attack in eval_attacks:
        if args.sor:
            args.batch_size = 1

        attacks = ",".join(args.attack_type)
        if args.model != args.apc_model:
            args.cross_model = True

        exp_name = f"{attacks}_{args.apc_model}"

        set_seed(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

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

        model = model.cuda()

        if 'add' in eval_attack.lower():
            # we add 512 points in adding attacks
            if args.num_points == 1024:
                num_points = 1024
                args.num_points = 1024 + 512
            elif args.num_points == 1024 + 512:
                num_points = 1024
        elif 'cluster' in eval_attack.lower():
            # we add 3*32=96 points in adding cluster attack
            if args.num_points == 1024:
                num_points = 1024
                args.num_points = 1024 + 3 * 32
            elif args.num_points == 1024 + 3 * 32:
                num_points = 1024
        else:
            num_points = 1024

        from config import BEST_WEIGHTS
        BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][num_points]
        if args.dataset == 'sonn':
            BEST_WEIGHTS[args.model] = BEST_WEIGHTS[args.model].replace(f"{args.model}/", f"{args.model}_{args.sonn_type}/")
        state_dict = torch.load(BEST_WEIGHTS[args.model], map_location='cpu')
        print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

        if args.dataset == 'mn40':
            test_dataset = ModelNet40APC(args, num_points=args.num_points, attack=eval_attack,
                                         partition='test', normalize=args.normalize_pc)
        elif args.dataset == 'sonn':
            test_dataset = ScanObjectNNAPC(args, num_points=args.num_points, attack=eval_attack,
                                           partition='test', normalize=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True,
                                 shuffle=False, num_workers=0)

        log_name = f'{exp_name.replace(args.model, f"{eval_attack}_{args.model}")}'
        if args.cross_model:
            log_name = f"xm_{log_name.replace(args.apc_model, f'{eval_attack}_{args.apc_model}_{args.model}')}"

        log_name = f"eval_{log_name}"
        logger = create_logger(f'./log/{args.dataset}/{exp_name}', log_name, 'info') if args.dataset == 'mn40' \
            else create_logger(f'./log/{args.sonn_type}/{exp_name}', log_name, 'info')
        logger.info(f'apc_model: {args.apc_model} <-> model: {args.model}')
        logger.info(f'Trained attack_type: {attacks} <-> attack_type_eval: {eval_attack}')
        if args.cross_model:
            logger.info('\t>>> Cross Model <<<')

        if args.apc_model == 'pt2':
            args.k = args.group_size
            prompter = APCT2R(args.k, args.feat_dims, args.off_dims, args).cuda()
        else:
            prompter = APC(args.k, args.feat_dims, args.off_dims, args).cuda()

        apc_path = f'log/{args.dataset}/{exp_name}/checkpoint_best.pt'
        if args.dataset == 'sonn':
            apc_path = apc_path.replace(args.dataset, args.sonn_type)
        prompter_state_dict = torch.load(apc_path)
        prompter.load_state_dict(prompter_state_dict['state_dict'])

        logger.info(f'>>> Loaded APC state from "{apc_path}"')
        logger.info(f'Best accuracy: {prompter_state_dict["best_acc1"]}')
        logger.info(f'Best epoch: {prompter_state_dict["epoch"]}')

        logger.info('>>> Evaluating... <<<')
        val_stats = evaluate(test_loader, model, prompter, eval_attack, logger)

        def_acc.append(val_stats['val_def_acc1'])

    print(f'>>> Loaded APC state from "{apc_path}"')
    print(f'Best accuracy: {prompter_state_dict["best_acc1"]}')
    print(f'Best epoch: {prompter_state_dict["epoch"]}')
    print(sum(def_acc) / len(def_acc), def_acc)


if __name__ == "__main__":
    main()
