# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:08:11 2021

@author: default
"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
device_ids = [0, 1]

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='ResNets for CIFAR10')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32', 
                    choices=model_names, 
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', 
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', 
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=-1, type=int, metavar='N', 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', 
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', 
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', 
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', 
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', 
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    
    # check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load model to multiple GPUs for training
    model = torch.nn.DataParallel(resnet.__dict__[args.arch](), device_ids = device_ids)
    # run model on GPUs
    model.cuda()
    cudnn.benchmark = True
    
    # load data using DataLoader and perform data augmentation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', 
                         train=True, 
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(), 
                             transforms.RandomCrop(32, 4), 
                             transforms.ToTensor(), 
                             normalize, 
                             ]), download=True), 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', 
                         train=False, 
                         transform=transforms.Compose([
                             transforms.ToTensor(), 
                             normalize, 
                             ])), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True)
    
    # define loss function and optimizer
    # This criterion combines LogSoftmax and NLLLoss in one single class.
    criterion = nn.CrossEntropyLoss().cuda()
    
    # 'half' method is used to reduce storage memory
    if args.half:
        model.half()
        criterion.half()
    
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr = args.lr, 
                                momentum = args.momentum, 
                                weight_decay = args.weight_decay, 
                                nesterov = True)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        milestones=[50, 100, 150, 200, 250], 
                                                        gamma=0.1)
    
    # optionally resume from a checkpoint
    # resume is a checkpoint path
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_state_dict'])
            
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    for epoch in range(args.start_epoch + 1, args.epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()
        
        # evaluate on the validation set
        prec1 = validate(val_loader, model, criterion)
        
        # remember best prec1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch, 
            'best_prec1': best_prec1, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_state_dict': lr_scheduler.state_dict(), 
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.pth.tar'))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_dir, 'model_best.pth.tar'))

def accuracy(output, target, topk=(1,)):
    # the shape of target is: [batch_size]
    # the shape of output is: [batch_size, classes]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # The shape of pred is: [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    # res contains top1 and top5 accuracy at the same time
    return res


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to train mode
    model.train()
    
    start_time = time.time()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for i, (input, target) in loop:
        # data loading time
        data_time.update(time.time() - start_time)
        
        # load data to GPU
        target_var = target.cuda()
        input_var = input.cuda()
        
        if args.half:
            input_var = input_var.half()
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        output = output.float()
        # the shape of output: [batch_size, num_classes]
        loss = loss.float()
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure accuracy and record loss
        # prec1 is top_1 accuracy
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        
        # update progress bar
        loop.set_description(f'Epoch [{epoch}]')
        loop.set_postfix(loss=losses.avg, acc=top1.avg, time=batch_time.avg)
    
    # save train accuracy and loss after each epoch
    log_folder = os.path.join(os.getcwd(), 'log')
    train_acc_f = os.path.join(log_folder, 'train_acc.txt')
    train_loss_f = os.path.join(log_folder, 'train_loss.txt')
    
    with open(train_acc_f, 'w') as f1:
        f1.write(str(top1.avg) + ' ')
    with open(train_loss_f, 'w') as f2:
        f2.write(str(losses.avg) + ' ')


    
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to evaluate model
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target_var = target.cuda()
            input_var = input.cuda()
            
            if args.half:
                input_var = input_var.half()
            
            output = model(input_var)
            loss = criterion(output, target_var)
            
            output = output.float()
            loss = loss.float()
            
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            
            # if i % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #               i, len(val_loader), batch_time=batch_time, loss=losses,
            #               top1=top1))

    print(' * Prec@1 {top1.avg:.4f}'
          .format(top1=top1))
    
    # save validation accuracy and loss after each epoch
    log_folder = os.path.join(os.getcwd(), 'log')
    val_acc_f = os.path.join(log_folder, 'val_acc.txt')
    val_loss_f = os.path.join(log_folder, 'val_loss.txt')
    
    with open(val_acc_f, 'w') as f3:
        f3.write(str(top1.avg) + ' ')
    with open(val_loss_f, 'w') as f4:
        f4.write(str(losses.avg) + ' ')
    
    return top1.avg

if __name__ == '__main__':
    main()

            
    
        
        
        
            
