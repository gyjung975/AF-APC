import os
import pdb
import time
import copy
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

from dataset import ModelNet40, ScanObjectNN
from model import DGCNN, PointNetCls, feature_transform_reguliarzer, PointNet2ClsSsg, PointTransformer
from torch.utils.data import DataLoader
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed
from tqdm import tqdm


def train(start_epoch):
    best_test_acc = 0
    best_acc_epoch = 0
    best_weight = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        step_count = 0
        all_loss_save = AverageMeter()
        if args.model.lower() == 'pointnet':
            loss_save = AverageMeter()
            fea_loss_save = AverageMeter()
        acc_save = AverageMeter()
        model.train()

        for data, label in train_loader:
            step_count += 1
            with torch.no_grad():
                data, label = data.float().cuda(), label.long().cuda()
                data = data.transpose(1, 2).contiguous()

            batch_size = data.size(0)
            opt.zero_grad()

            if args.model.lower() == 'pointnet':
                logits, trans, trans_feat = model(data)
                loss = criterion(logits, label, False)
                # we may need to calculate feature_transform loss
                if args.feature_transform:
                    fea_loss = feature_transform_reguliarzer(trans_feat) * 0.001
                else:
                    fea_loss = torch.tensor(0.).cuda()
                all_loss = loss + fea_loss
                all_loss.backward()
                opt.step()

                acc = (torch.argmax(logits, dim=-1) == label).sum().float() / float(batch_size)

                # statistics accumulation
                all_loss_save.update(all_loss.item(), batch_size)
                loss_save.update(loss.item(), batch_size)
                fea_loss_save.update(fea_loss.item(), batch_size)
                acc_save.update(acc.item(), batch_size)
                if step_count % args.print_iter == 0:
                    print('Epoch {}, step {}, lr: {:.6f}\n'
                          'All loss: {:.4f}, loss: {:.4f}, Fea loss: {:.4f}\n'
                          'Train acc: {:.4f}'.
                          format(epoch, step_count, get_lr(opt),
                                 all_loss_save.avg, loss_save.avg,
                                 fea_loss_save.avg, acc_save.avg))
            else:
                logits = model(data)
                all_loss = criterion(logits, label, False)
                all_loss.backward()
                opt.step()

                acc = (torch.argmax(logits, dim=-1) == label).sum().float() / float(batch_size)

                all_loss_save.update(all_loss.item(), batch_size)
                acc_save.update(acc.item(), batch_size)
                if step_count % args.print_iter == 0:
                    print('Epoch {}, step {}, lr: {:.6f}\n'
                          'All loss: {:.4f}, train acc: {:.4f}'.
                          format(epoch, step_count, get_lr(opt),
                                 all_loss_save.avg, acc_save.avg))
                    torch.cuda.empty_cache()

        if epoch % 10 == 0 or epoch > 180:
            acc = test()
            if acc > best_test_acc:
                best_test_acc = acc
                best_acc_epoch = epoch
                best_weight = copy.deepcopy(model.state_dict())

            print('Epoch {}, acc {:.4f}\nCurrent best acc {:.4f} at epoch {}'.
                  format(epoch, acc, best_test_acc, best_acc_epoch))
            torch.save(model.state_dict(),
                       os.path.join(logs_dir, 'model{}_acc_{:.4f}_loss_{:.4f}_lr_{:.6f}.pth'.
                                    format(epoch, acc, all_loss_save.avg, get_lr(opt))))
            torch.cuda.empty_cache()
            logger.add_scalar('test/acc', acc, epoch)

        logger.add_scalar('train/loss', all_loss_save.avg, epoch)
        logger.add_scalar('train/lr', get_lr(opt), epoch)
        scheduler.step(epoch)

    torch.save(best_weight, os.path.join(logs_dir, 'BEST_model{}_acc_{:.4f}.pth'.format(best_acc_epoch, best_test_acc)))


def test():
    model.eval()
    acc_save = AverageMeter()
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.float().cuda(), label.long().cuda()
            data = data.transpose(1, 2).contiguous()
            batch_size = data.size(0)
            logits = model(data)[0]
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == label).sum().float() / float(batch_size)
            acc_save.update(acc.item(), batch_size)

    print('Test accuracy: {:.4f}'.format(acc_save.avg))
    return acc_save.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pt2'])
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--trans_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--num_group', type=int, default=64)
    parser.add_argument('--encoder_dims', type=int, default=256)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='mn40',
                        choices=['mn40', 'sonn'])
    parser.add_argument('--sonn_type', type=str, default='obj_only',
                        choices=['obj_only', 'obj_bg', 'hardest'])
    parser.add_argument('--batch_size', type=int, default=256, help='Size of batch')
    parser.add_argument('--epochs', type=int, default=300, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    set_seed(1)
    print(args)

    ncls = 40 if args.dataset == 'mn40' else 15
    if args.model.lower() == 'dgcnn':
        model = DGCNN(output_channels=ncls, emb_dims=args.emb_dims, k=args.k)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=ncls, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=ncls)
    elif args.model.lower() == 'pt2':
        model = PointTransformer(args, num_classes=ncls, pretrained=args.pretrained)
    else:
        print('Model not recognized')
        exit(-1)

    model = nn.DataParallel(model).cuda()

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    if args.dataset == 'mn40':
        data_root = "data/modelnet40_normal_resampled/modelnet40_{}_1024pts.dat"
        train_set = ModelNet40(args, data_root.format('train'), num_points=args.num_points,
                               normalize=True, partition='train', victim=True)
        test_set = ModelNet40(args, data_root.format('test'), num_points=args.num_points,
                              normalize=True, partition='test', victim=True)
    elif args.dataset == 'sonn':
        train_set = ScanObjectNN(args, sonn_type=args.sonn_type, num_points=args.num_points,
                                 normalize=False, partition='train', victim=True)
        test_set = ScanObjectNN(args, sonn_type=args.sonn_type, num_points=args.num_points,
                                normalize=False, partition='test', victim=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    criterion = cal_loss

    start_datetime = time.strftime("%Y-%m-%d", time.localtime())
    logs_dir = "pretrain/{}/{}/{}_{}".format(args.dataset, args.model, start_datetime, args.num_points)
    if args.dataset == 'sonn':
        logs_dir = logs_dir.replace(args.model, f"{args.model}_{args.sonn_type}")
    os.makedirs(logs_dir, exist_ok=True)
    logger = SummaryWriter(os.path.join(logs_dir, 'logs'))

    train(1)
