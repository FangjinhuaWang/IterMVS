import argparse
import os
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='IterMVS for high-resolution multi-view stereo')
parser.add_argument('--mode', default='train', help='train or val', choices=['train', 'val'])
parser.add_argument('--model', default='IterMVS', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--valpath', help='validation datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--vallist', help='validation list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="4,8,12:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--regress', action='store_true', help='train the regression and confidence')
parser.add_argument('--small_image', action='store_true', help='train with small input as 640x512, otherwise train with 1280x1024')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')


parser.add_argument('--iteration', type=int, default=4, help='num of iteration of GRU')




# parse arguments and check
args = parser.parse_args()
if args.resume: # store_true means set the variable as "True"
    assert args.mode == "train"
    assert args.loadckpt is None
if args.valpath is None:
    args.valpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)

train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 5, robust_train=True)
test_dataset = MVSDataset(args.valpath, args.vallist, "val", 5,  robust_train=False)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = Pipeline(iteration=args.iteration)
if args.mode in ["train", "val"]:
    model = nn.DataParallel(model)
model.cuda()
model_loss = full_loss
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)


# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "val" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)
    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            do_summary_image = global_step % (50*args.summary_freq) == 0
            
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            if do_summary_image:
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), loss,
                                                                                     time.time() - start_time))
        lr_scheduler.step()

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # testing
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            do_summary_image = global_step % (50*args.summary_freq) == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
            if do_summary_image:
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"] 
    mask = sample_cuda["mask"]      
    depth_gt_0 = depth_gt['level_0']
    mask_0 = mask['level_0']
    depth_gt_1 = depth_gt['level_2']
    mask_1 = mask['level_2']

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"])

    depth_est = outputs["depths"]
    confidences_est = outputs["confidences"]
    depth_upsampled_est = outputs["depths_upsampled"]
    loss = model_loss(depth_est, depth_upsampled_est, confidences_est, depth_gt, mask, sample_cuda["depth_min"], sample_cuda["depth_max"], args.regress)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    scalar_outputs = {"loss": loss}
    image_outputs = {
                        "depth_gt": depth_gt_1 * mask_1,
                        "depth_initial": depth_est["combine"][0] * mask_1,
                        "ref_img": sample["imgs"]['level_2'][:, 0],
                        "depth_final_full": depth_upsampled_est[-1] * mask_0
                        }
    if detailed_summary:
        image_outputs["errormap_initial"] = (depth_est["combine"][0] - depth_gt_1).abs() * mask_1
        image_outputs["errormap_final_full"] = (depth_upsampled_est[-1] - depth_gt_0).abs() * mask_0

    scalar_outputs["abs_error_initial"] = AbsDepthError_metrics(depth_est["combine"][0], depth_gt_1, mask_1 > 0.5)
    scalar_outputs["thres1mm_initial"] = Thres_metrics(depth_est["combine"][0], depth_gt_1, mask_1 > 0.5, 1)
    scalar_outputs["abs_error_final_full"] = AbsDepthError_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5)
    # threshold = 1mm
    scalar_outputs["thres1mm_final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 1)
    # threshold = 2mm
    scalar_outputs["thres2mm_final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 2)
    # threshold = 4mm
    scalar_outputs["thres4mm_final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 4)
    # threshold = 8mm
    scalar_outputs["thres8mm_final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 8)

    iters = args.iteration+1
    for j in range(1, iters):
        scalar_outputs[f"thres1mm_gru_{j}"] = Thres_metrics(depth_est["combine"][j], depth_gt_1, mask_1 > 0.5, 1)
        scalar_outputs[f"abs_error_gru_{j}"] = AbsDepthError_metrics(depth_est["combine"][j], depth_gt_1, mask_1 > 0.5)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"] 
    mask = sample_cuda["mask"]      
    depth_gt_0 = depth_gt['level_0']
    mask_0 = mask['level_0']
    depth_gt_1 = depth_gt['level_2']
    mask_1 = mask['level_2']

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"])

    depth_est = outputs["depths"]
    confidences_est = outputs["confidences"]
    depth_upsampled_est = outputs["depths_upsampled"]
    loss = model_loss(depth_est, depth_upsampled_est, confidences_est, depth_gt, mask, sample_cuda["depth_min"], sample_cuda["depth_max"], args.regress)
    scalar_outputs = {"loss": loss}
    image_outputs = {
                    "depth_gt": depth_gt_1 * mask_1,
                    "depth_initial": depth_est["combine"][0] * mask_1,
                    "ref_img": sample["imgs"]['level_2'][:, 0],
                    "depth_final_full": depth_upsampled_est[-1] * mask_0
                    }

    if detailed_summary:
        image_outputs["errormap_initial"] = (depth_est["combine"][0] - depth_gt_1).abs() * mask_1
        image_outputs["errormap_final_full"] = (depth_upsampled_est[-1] - depth_gt_0).abs() * mask_0
        

    scalar_outputs["abs_error_initial"] = AbsDepthError_metrics(depth_est["combine"][0], depth_gt_1, mask_1 > 0.5)
    scalar_outputs["thres1mm_initial"] = Thres_metrics(depth_est["combine"][0], depth_gt_1, mask_1 > 0.5, 1)
    scalar_outputs["abs_error_final_full"] = AbsDepthError_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5)
    # threshold = 1mm
    scalar_outputs["thres1mm_final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 1)
    # threshold = 2mm
    scalar_outputs["thres2mm_final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 2)
    # threshold = 4mm
    scalar_outputs["thres4mm_final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 4)
    # threshold = 8mm
    scalar_outputs["thres8mm_final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 8)

    iters = args.iteration+1
    for j in range(1, iters):
        scalar_outputs[f"thres1mm_gru_{j}"] = Thres_metrics(depth_est["combine"][j], depth_gt_1, mask_1 > 0.5, 1)
        scalar_outputs[f"abs_error_gru_{j}"] = AbsDepthError_metrics(depth_est["combine"][j], depth_gt_1, mask_1 > 0.5)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "val":
        test()
    
