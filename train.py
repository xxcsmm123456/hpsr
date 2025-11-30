import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import random
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from data.option import parser
from model.HPSR import HPSRnet
from data.dataset import HPSRdataset
from data.utils import AverageMeter, calc_psnr, save_checkpoint
import os
import shutil
import logging
import math
from torch.nn import DataParallel
logging.basicConfig(filename='train_0304.log',format='[%(message)s]', level = logging.DEBUG,filemode='a',datefmt='%Y-%m-%d%I:%M:%S %p')
os.environ['CUDA_VISIBLE_DEVICES']='0'

def train(dataset, loader, model, criterion, optimizer, tag=''):
    losses = AverageMeter()
    psnr = AverageMeter()
    # Set the model to training mode
    model.train()
    with tqdm(total=len(dataset)) as t:
        t.set_description(tag)
        for data in loader:
            lr,dem,CLCD,pre,tmp, hr = data
            if torch.cuda.is_available():
                lr = lr.cuda()
                dem = dem.cuda()
                hr = hr.cuda()
                CLCD = CLCD.cuda()
                pre = pre.cuda()
                tmp = tmp.cuda()
            sr = model(lr,dem,CLCD,pre,tmp)
            loss = criterion(sr, hr)
            losses.update(loss.item(), lr.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            psnr.update(calc_psnr(sr, hr, scale=args.scale, max_value=1), lr.shape[0])#args.rgb_range[1]
            t.set_postfix(loss='{:.4f}'.format(losses.avg),psnrs='{:.4f}'.format(psnr.avg))
            t.update(lr.shape[0])
        return losses.avg,psnr.avg

def test(dataset, loader, model, criterion, args, tag=''):
    losses = AverageMeter()
    psnr = AverageMeter()
    # Set the model to evaluation mode
    model.eval()
    with tqdm(total=len(dataset)) as t:
        t.set_description(tag)
        for data in loader:
            lr,dem,CLCD,pre,tmp, hr = data
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
                dem = dem.cuda()
                CLCD = CLCD.cuda()
                pre = pre.cuda()
                tmp = tmp.cuda()
            with torch.no_grad():
                sr = model(lr,dem,CLCD,pre,tmp)
            loss = criterion(sr, hr)
            losses.update(loss.item(), lr.shape[0])
            psnr.update(calc_psnr(sr, hr, scale=args.scale, max_value=1), lr.shape[0])#args.rgb_range[1]
            t.set_postfix(loss='{:.4f}'.format(losses.avg),psnrs='{:.4f}'.format(psnr.avg))
            t.update(lr.shape[0])

        return losses.avg, psnr.avg

if __name__ == '__main__':
    # Define specific options and parse arguments
    parser.add_argument('--dataset-dir', default=r'datasets',type=str,  help='Dataset Directory')
    parser.add_argument('--output-dir', default=r'/output/',type=str)
    args = parser.parse_args()
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = HPSRnet(args)
    criterion = nn.L1Loss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion=criterion.cuda()

    train_dataset = HPSRdataset('./temp','train', upscale_factor = args.scale, input_transform = transforms.ToTensor(),
                                  target_transform = transforms.ToTensor())
    valid_dataset = HPSRdataset('./temp','val', upscale_factor = args.scale, input_transform = transforms.ToTensor(),
                                target_transform = transforms.ToTensor())
    train_dataloader = DataLoader(dataset = train_dataset, num_workers = 8, batch_size = args.batch_size, shuffle = True)
    valid_dataloader = DataLoader(dataset = valid_dataset, num_workers = 8, batch_size = args.batch_size, shuffle = False)
    best_epoch = 0
    best_loss = 0
    best_psnr = 0
    output_name = '{}-f{}-b{}-r{}-x{}'.format(args.model, args.n_feats, args.n_res_blocks, args.expansion_ratio, args.scale)
    checkpoint_name = '{}-latest.pth'.format(output_name)
    from torch.optim import lr_scheduler
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for epoch in range(args.epochs):
        print('[epoch: {}/{}]'.format(epoch + 1, args.epochs))
        # Train
        train_loss,train_psnr = train(train_dataset, train_dataloader, model, criterion, optimizer, tag='train')
        logging.info('[Epoch %d] Train Loss: %.4f (PSNR: %.4f db)' % (epoch, train_loss, train_psnr))
        # Validate
        valid_loss, valid_psnr = test(valid_dataset, valid_dataloader, model, criterion, args, tag='valid')
        logging.info('[Epoch %d] Val Loss: %.4f (PSNR: %.4f db)' % (epoch, valid_loss, valid_psnr))
        is_best = valid_psnr > best_psnr
        if valid_psnr > best_psnr:
            best_epoch = epoch
            best_loss = valid_loss
            best_psnr = valid_psnr
        logging.info('* PSNR: {:.4f}'.format(valid_psnr))
        logging.info('* best PSNR: {:.4f} @ epoch: {}\n'.format(best_psnr, best_epoch + 1))
        # Save checkpoint
        checkpoint_name = '{}-{}-{:.2f}-latest.pth'.format(output_name,epoch,valid_psnr)
        save_checkpoint({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_psnr': train_psnr,
            'valid_loss': valid_loss,
            'valid_psnr': valid_psnr,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.output_dir, checkpoint_name), is_best)

        exp_lr_scheduler.step()
