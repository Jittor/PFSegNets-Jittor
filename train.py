from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import numpy as np
import jittor as jt

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist, set_bn_eval
from utils.f_boundary import eval_mask_boundary
import datasets
import loss
import network
import optimizer

jt.flags.use_cuda = 1

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')

parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')
parser.add_argument('--dice_loss', default=False,
                    action='store_true', help="whether use dice loss in edge")
parser.add_argument("--ohem", action="store_true",
                    default=False, help="start OHEM loss")
parser.add_argument("--aux", action="store_true",
                    default=False, help="whether use Aux loss")
parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--joint_edge_loss_pfnet', action='store_true')
parser.add_argument('--edge_weight', type=float, default=1.0,
                    help='Edge loss weight for joint loss')
parser.add_argument('--seg_weight', type=float, default=1.0,
                    help='Segmentation loss weight for joint loss')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_epoch', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=True,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=4,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=1.0,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=1.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--fix_bn', action='store_true', default=False,
                    help=" whether to fix bn for improving the performance")
parser.add_argument('--evaluateF', action='store_true', default=False,
                    help="whether to evaluate the F score")
parser.add_argument('--eval_thresholds', type=str, default='0.0005,0.001875,0.00375,0.005',
                    help='Thresholds for boundary evaluation')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--print_freq', type=int, default=5,
                    help='frequency of print')
parser.add_argument('--eval_freq', type=int, default=1,
                    help='frequency of evaluation during training')
parser.add_argument('--with_aug', action='store_true')
parser.add_argument('--thicky', default=8, type=int)
parser.add_argument('--draw_train', default=False, action='store_true')
parser.add_argument('--match_dim', default=64, type=int,
                    help='dim when match in pfnet')
parser.add_argument('--ignore_background', action='store_true', help='whether to ignore background class when '
                                                                     'generating coarse mask in pfnet')
parser.add_argument('--maxpool_size', type=int, default=9)
parser.add_argument('--avgpool_size', type=int, default=9)
parser.add_argument('--edge_points', type=int, default=32)
parser.add_argument('--start_epoch', type=int, default=0)
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}


# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2


def main():
    """
    Main Function
    """

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    train_loader, val_loader = datasets.setup_loaders(args)
    criterion, criterion_val = loss.get_loss(args)
    net = network.get_net(args, criterion)
    optim, scheduler = optimizer.get_optimizer(args, net)

    if args.fix_bn:
        net.apply(set_bn_eval)
        print("Fix bn for finetuning")

    if args.snapshot:
        optimizer.load_weights(net, optim,
                               args.snapshot, args.restore_optimizer)
    if args.evaluateF:
        assert args.snapshot is not None, "must load weights for evaluation"
        evaluate(val_loader, net, args)
        return
    # Main Loop
    for epoch in range(args.start_epoch, args.max_epoch):
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.EPOCH = epoch
        cfg.immutable(True)

        scheduler.step()
        train(train_loader, net, optim, epoch, writer)
        if epoch % args.eval_freq == 0 or epoch == args.max_epoch - 1:
            validate(val_loader, net, criterion_val,
                     optim, epoch, writer)


def train(train_loader, net, optim, curr_epoch, writer):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    train_main_loss = AverageMeter()
    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        edges = None
        if args.joint_edge_loss_pfnet:
            inputs, gts, edges, _img_name = data
        else:
            inputs, gts, _img_name = data

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        optim.zero_grad()
        if args.joint_edge_loss_pfnet:
            main_loss_dic = net(inputs, gts=(gts, edges))
            main_loss = 0.0
            for v in main_loss_dic.values():
                main_loss = main_loss + v
        else:
            main_loss = net(inputs, gts=gts)
        main_loss = main_loss.mean()
        log_main_loss = main_loss.clone()

        train_main_loss.update(log_main_loss.item(), batch_pixel_size)

        optim.backward(main_loss)

        optim.step()

        curr_iter += 1

        if i % args.print_freq == 0:
            if args.joint_edge_loss_pfnet:
                msg = f'[epoch {curr_epoch}], [iter {i + 1} / {len(train_loader)}], '
                msg += '[seg_main_loss:{:0.5f}]'.format(
                    main_loss_dic['seg_loss'])
                for j in range(3):
                    temp_msg = '[layer{}:, [edge loss {:0.5f}] '.format(
                        (3-j), main_loss_dic[f'edge_loss_layer{3-j}'])
                    msg += temp_msg
                msg += ', [lr {:0.5f}]'.format(optim.param_groups[-1]['lr'])
            else:
                msg = '[epoch {}], [iter {} / {}], [train main loss {:0.6f}], [lr {:0.6f}]'.format(
                    curr_epoch, i + 1, len(train_loader), train_main_loss.avg,
                    optim.param_groups[-1]['lr'])

            logging.info(msg)
            print(msg, flush=True)

            # Log tensorboard metrics for each iteration of the training phase
            writer.add_scalar('training/loss', (train_main_loss.val),
                              curr_iter)
            writer.add_scalar('training/lr', optim.param_groups[-1]['lr'],
                              curr_iter)

        if i > 5 and args.test_mode:
            return


@jt.single_process_scope()
def validate(val_loader, net, criterion, optim, curr_epoch, writer):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        inputs, gt_image, img_names = data
        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        with jt.no_grad():
            output = net(inputs)

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == args.dataset_cls.num_classes

        val_loss.update(criterion(output, gt_image).item(), batch_pixel_size)
        predictions = output.argmax(dim=1)[0]      # prediction map

        # Logging
        if val_idx % 20 == 0:
            logging.info("validating: %d / %d",
                         val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             args.dataset_cls.num_classes)
        del output, val_idx, data

    evaluate_eval(args, net, optim, val_loss, iou_acc, dump_images,
                  writer, curr_epoch, args.dataset_cls)

    return val_loss.avg


def evaluate(val_loader, net, args):
    '''
    Runs the evaluation loop and prints F score
    val_loader: Data loader for validation
    net: thet network
    return:
    '''
    net.eval()
    for i, thresh in enumerate(args.eval_thresholds.split(',')):
        Fpc = np.zeros((args.dataset_cls.num_classes))
        Fc = np.zeros((args.dataset_cls.num_classes))
        val_loader.sampler.set_epoch(i + 1)
        evaluate_F_score(val_loader, net, thresh, Fpc, Fc)


def evaluate_F_score(val_loader, net, thresh, Fpc, Fc):
    for vi, data in enumerate(val_loader):
        input, mask, img_names = data
        assert len(input.size()) == 4 and len(mask.size()) == 3
        assert input.size()[2:] == mask.size()[1:]

        with jt.no_grad():
            seg_out = net(input)

        seg_predictions = seg_out.data.max(1)[1].cpu()

        print('evaluating: %d / %d' % (vi + 1, len(val_loader)))
        _Fpc, _Fc = eval_mask_boundary(seg_predictions.numpy(), mask.numpy(), args.dataset_cls.num_classes,
                                       bound_th=float(thresh))
        Fc += _Fc
        Fpc += _Fpc

        del seg_out, vi, data

    logging.info('Threshold: ' + thresh)
    logging.info('F_Score: ' + str(np.sum(Fpc / Fc) /
                                   args.dataset_cls.num_classes))
    logging.info('F_Score (Classwise): ' + str(Fpc / Fc))

    return Fpc


if __name__ == '__main__':
    main()
