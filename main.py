import argparse
import os
import shutil
import time
import copy
import PIL.Image as Image
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from datasets import get_loader
from losses import get_loss
from models import get_model
from utils import get_scheduler, get_optimizer, accuracy, save_checkpoint, AverageMeter


parser = argparse.ArgumentParser(description='Self-Adaptive Trainingn')
# network
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    help='model architecture')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--base-width', default=64, type=int,
                    help='base width of resnets or hidden dim of fc nets')
# training setting
parser.add_argument('--data-root', help='The directory of data',
                    default='~/datasets/CIFAR10', type=str)
parser.add_argument('--dataset', help='dataset used to training',
                    default='cifar10', type=str)
parser.add_argument('--train-sets', help='subsets (train/trainval) that used to training',
                    default='train', type=str)
parser.add_argument('--val-sets', type=str, nargs='+', default=['noisy_val'],
                    help='subsets (clean_train/noisy_train/clean_val/noisy_val) that used to validation')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='optimizer for training')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-schedule', default='step', type=str,
                    help='LR decay schedule')
parser.add_argument('--lr-milestones', type=int, nargs='+', default=[40, 80],
                    help='LR decay milestones for step schedule.')
parser.add_argument('--lr-gamma', default=0.1, type=float,
                    help='LR decay gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# noisy setting
parser.add_argument('--noise-rate', default=0., type=float,
                    help='Label noise rate')
parser.add_argument('--noise-type', default=None, type=str,
                    help='Noise type, could be one of (corrupted_label, Gaussian, random_pixels, shuffled_pixels)')
parser.add_argument('--noise-info', default=None, type=str, 
                    help='directory of pre-configured noise pattern.')
parser.add_argument('--use-refined-label', action='store_true', help='whether or not use refined label by self-adaptive training')
parser.add_argument('--turn-off-aug', action='store_true', help='whether or not use data augmentation')
# loss function
parser.add_argument('--loss', default='ce', help='loss function')
parser.add_argument('--sat-alpha', default=0.9, type=float,
                    help='momentum term of self-adaptive training')
parser.add_argument('--sat-es', default=0, type=int,
                    help='start epoch of self-adaptive training (default 0)')
# misc
parser.add_argument('-s', '--seed', default=None, type=int,
                    help='number of data loading workers (default: None)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-freq', default=0, type=int,
                    help='print frequency (default: 0, i.e., only best and latest checkpoints are saved)')

#MIXUP:
parser.add_argument('--mixup', action='store_true',
                    help='Whether or not to use mixup')
parser.add_argument('--mixup-alpha', default=1, type=float,
                    help='Value of alpha in mixup.')
parser.add_argument('--mixup-gamma', default=.1, type=float,
                    help='Value of gamma in mixup.')

parser.add_argument('--is_tpu', action='store_true',
                    help='Whether or not we are on a TPU')

parser.add_argument('--aggressive', action='store_true',
                    help='Update Labels aggressively, as opposed to mixing it up.')
args = parser.parse_args()

if args.is_tpu:
    import torch_xla.core.xla_model as xm

best_prec1 = 0
if args.seed is None:
    import random
    args.seed = random.randint(1, 10000)

def main():
    ## dynamically adjust hyper-parameters for ResNets according to base_width
    if args.base_width != 64 and 'sat' in args.loss:
        factor = 64. / args.base_width
        args.sat_alpha = args.sat_alpha**(1. / factor)
        args.sat_es = int(args.sat_es * factor)
        print("Adaptive parameters adjustment: alpha = {:.3f}, Es = {:d}".format(args.sat_alpha, args.sat_es))

    print(args)
    global best_prec1

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # prepare dataset
    train_loader, val_loaders, test_loader, num_classes, targets, clean_targets = get_loader(args)
    if args.is_tpu:
        device = xm.xla_device()
    
    
    model = get_model(args, num_classes, base_width=args.base_width)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.is_tpu:
        model = model.to(device)
    else:
        model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    criterion = get_loss(args, device=device, labels=targets, num_classes=num_classes)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    if args.evaluate:
        validate(test_loader, model, device)
        return

    print("*" * 40)
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)
        print("*" * 40)

        # evaluate on validation sets
        prec1 = 0
        for name, val_loader in zip(args.val_sets, val_loaders):
            print(name +":", end="\t")
            prec1 = validate(val_loader, model, device, epoch)
        print("*" * 40)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            filename = 'checkpoint_{}.tar'.format(epoch + 1)
        else:
            filename = None
        save_checkpoint(args.save_dir, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=filename)
    
    # evaluate latest checkpoint
    print("Test acc of latest checkpoint:", end='\t')
    validate(test_loader, model, device)
    print("*" * 40)

    # evaluate best checkpoint
    if len(val_loaders) > 0:
        checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint_best.tar'))
        print("Best validation acc ({}th epoch): {}".format(checkpoint['epoch'], best_prec1))
        model.load_state_dict(checkpoint['state_dict'])
        print("Test acc of best checkpoint:", end='\t')
        validate(test_loader, model, device)
        print("*" * 40)

    # save soft label
    if hasattr(criterion, 'soft_labels'):
        out_fname = os.path.join(args.save_dir, 'updated_soft_labels.npy')
        np.save(out_fname, criterion.soft_labels.cpu().numpy())
        print("Updated soft labels is saved to {}".format(out_fname))

    # save noise targets
    out_fname = os.path.join(args.save_dir, 'noisy_labels.npy')
    np.save(out_fname, targets)
    print("Noisy labels saved to {}".format(out_fname))

    # save clean targets
    out_fname = os.path.join(args.save_dir, 'clean_labels.npy')
    np.save(out_fname, clean_targets)
    print("Clean labels saved to {}".format(out_fname))

def mixup_data(x, y, index, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    indices = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[indices, :]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b, lam, index, index[indices]

def mixup_criterion(i, criterion, pred, y_a, y_b, lam, index1, index2, epoch, clean_out, args):
    if args.aggressive:
        first = lam * criterion(pred, y_a, index1, epoch, clean_out, True, args)
        second = (1 - lam) * criterion(pred, y_b, index2, epoch, clean_out, False, args)
    else:
        first = lam * criterion(pred, y_a, index1, epoch, clean_out, lam > 1 - args.mixup_gamma, args)
        second = (1 - lam) * criterion(pred, y_b, index2, epoch, clean_out, lam < args.mixup_gamma, args)
    if i == 175:
        print(epoch, lam, first, second)
    return first + second

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)
        index = index.to(device)
        if args.mixup:
            mixed_input, target1, target2, lambdas, index1, index2 = mixup_data(input, target, index, args.mixup_alpha, device)
        # compute output
        if args.mixup:
            output = model(mixed_input)
            clean_out = model(input)
            loss = mixup_criterion(i, criterion, output, target1, target2, lambdas, index1, index2, epoch, clean_out, args)
        else:
            output = model(input)
            loss = criterion(output, target, index, epoch, output, True, args)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.is_tpu:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(train_loader):
            lr = optimizer.param_groups[0]['lr']
            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR {lr:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i+1, len(train_loader), lr=lr, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, device, epoch=0):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    all_targets = []
    all_logits = []
    for i, (input, target, _) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        all_targets.append(target)
        # compute output
        with torch.no_grad():
            output = model(input)
            all_logits.append(output)
            loss = F.cross_entropy(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    main()
