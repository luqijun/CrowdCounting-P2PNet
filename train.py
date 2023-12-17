import argparse
import datetime
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--match_type', default='all', type=str,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./new_public_density_data',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--vis_dir', default='',
                        help='path where to save vis images, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_start', default=300, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # create the logging file
    sub_dir = datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M%S')  # prepare saving path
    log_dir = os.path.join(args.output_dir, sub_dir)
    os.makedirs(log_dir, exist_ok=True)
    vis_dir = None
    if args.vis_dir:
        vis_dir = os.path.join(log_dir, args.vis_dir)
        os.makedirs(vis_dir, exist_ok=True)

    run_log_name = os.path.join(log_dir, 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        # log_file.write("{}\n".format(args))
        for k, v in args.__dict__.items():  # save args
            log_file.write("{}: {} \n".format(k, v))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the P2PNet model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set = loading_data(args.data_root)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    # fix bug: RuntimeError: received 0 items of ancdata
    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume the weights and training state if exists
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    # the logger writer
    writer = SummaryWriter(log_dir)

    best_mae = 1e9
    best_epoch = 0
    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
                log_file.write("loss/loss_ce@{}: {}\n".format(epoch, stat['loss_ce']))
                
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)


        # change lr according to the scheduler
        lr_scheduler.step()
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'saved/latest.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
        }, checkpoint_latest_path)
        # run evaluation
        if epoch > args.eval_start and epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(model, data_loader_val, device, vis_dir)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])

            if result[0] < best_mae:
                best_mae = result[0]
                best_epoch = epoch

            # print the evaluation results
            print('=======================================test=======================================')
            print("val-- ", "mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", best_mae, "best epoch:", best_epoch,)
            with open(run_log_name, "a") as log_file:
                log_file.write("val-- mae:{}, mse:{}, time:{}, best mae:{}, best epoch:{} \n".format(result[0],
                                result[1], t2 - t1, best_mae, best_epoch))
            print('=======================================test=======================================')
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, result[0]))
                    log_file.write("metric/mse@{}: {} \n".format(step, result[1]))
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

            # save the best model since begining
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'saved/best_mae.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_best_path)
    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)