# python run.py gaofen/img/val/image test_img/
import os
import sys
import argparse
from datetime import datetime
import cv2
from PIL import Image
import jittor as jt
import jittor.transform as transforms
from jittor import nn
import numpy as np
from config import assert_and_infer_cfg
from optimizer import restore_snapshot
import network
from datasets import GAOFENIMG

jt.flags.use_cuda = 1

input_path = sys.argv[1]
output_path = sys.argv[2]
del sys.argv[1]
del sys.argv[1]

print(input_path, output_path)

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dump_images', action='store_true', default=False)
parser.add_argument('--arch', type=str, default='')
parser.add_argument('--single_scale', action='store_true', default=False)
parser.add_argument('--scales', type=str, default='0.5,1.0,2.0')
parser.add_argument('--dist_bn', action='store_true', default=False)
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--fixed_aspp_pool', action='store_true', default=False,
                    help='fix the aspp image-level pooling size to 105')
parser.add_argument('--dataset_cls', type=str,
                    default='cityscapes', help='cityscapes')
parser.add_argument('--sliding_overlap', type=float, default=1 / 3)
parser.add_argument('--no_flip', action='store_true', default=False,
                    help='disable flipping')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, video_folder')
parser.add_argument('--trunk', type=str, default='resnet101', help='cnn trunk')
parser.add_argument('--dataset_dir', type=str, default=None,
                    help='Dataset Location')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--crop_size', type=int, default=513)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (4 items evaluated) to verify nothing failed')
parser.add_argument('--cv_split', type=int, default=None)
parser.add_argument('--mode', type=str, default='fine')
parser.add_argument('--split_index', type=int, default=0)
parser.add_argument('--split_count', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume Inference')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Only in pooling mode')
parser.add_argument('--maxpool_size', type=int, default=9)
parser.add_argument('--avgpool_size', type=int, default=9)
parser.add_argument('--edge_points', type=int, default=32)
parser.add_argument('--match_dim', default=64, type=int,
                    help='dim when match in pfnet')
parser.add_argument('--ignore_background', action='store_true', help='whether to ignore background class when '
                                                                     'generating coarse mask in pfnet')
parser.add_argument('--input_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)

args = parser.parse_args()
args.input_path = input_path
args.output_path = output_path
args.dump_images = True
args.arch = 'network.pointflow_resnet_with_max_avg_pool.DeepR2N101_PF_maxavg_deeply'
args.single_scale = True
args.scales = 1.0
args.cv_split = 0

args.maxpool_size = 14
args.avgpool_size = 9
args.edge_points = 128
args.match_dim = 64

args.no_flip = False
args.dataset_cls = GAOFENIMG
args.snapshot = 'last_epoch_63_mean-iu_0.97659.pkl'

assert_and_infer_cfg(args, train_mode=False)
mean_std = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))


def inference_whole(model, img, scales):
    """
        whole images inference
    """
    origw, origh = img.size
    preds = []
    if args.no_flip:
        flip_range = 1
    else:
        flip_range = 2

    for scale in scales:
        target_w, target_h = int(512 * scale), int(512 * scale)
        scaled_img = img.resize((target_w, target_h), Image.BILINEAR)

        for flip in range(flip_range):
            if flip:
                scaled_img = scaled_img.transpose(Image.FLIP_LEFT_RIGHT)

            img_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.ImageNormalize(*mean_std)])
            image = img_transform(scaled_img)
            with jt.no_grad():
                input = jt.Var(image).unsqueeze(0)
                scale_out = model(input)
                scale_out = nn.upsample(scale_out, size=(
                    origh, origw), mode="bilinear", align_corners=True)
                scale_out = scale_out.squeeze(0).numpy()
                if flip:
                    scale_out = scale_out[:, :, ::-1]
            preds.append(scale_out)

    return preds


def get_net():
    """
    Get Network for evaluation
    """
    print('Load model file: %s', args.snapshot)
    print(args)
    net = network.get_net(args, criterion=None)
    net, _ = restore_snapshot(net, optimizer=None,
                              snapshot=args.snapshot, restore_optimizer_bool=False)
    net.eval()
    return net


class RunEval():
    def __init__(self, output_dir, write_image):
        self.pred_path = output_dir

        self.write_image = write_image
        self.time_list = []
        self.mapping = {}
        os.makedirs(self.pred_path, exist_ok=True)

    def inf(self, imgs, img_names, net, scales):
        self.img_name = img_names
        pred_img_name = '{}/{}.png'.format(self.pred_path, self.img_name)

        prediction_pre_argmax_collection = inference_whole(net, imgs, scales)

        prediction_pre_argmax = np.mean(
            prediction_pre_argmax_collection, axis=0)
        prediction = np.argmax(prediction_pre_argmax, axis=0)
        if self.write_image:
            cv2.imwrite(pred_img_name, prediction*255)


def main():
    if args.single_scale:
        scales = [1.0]
    else:
        scales = [float(x) for x in args.scales.split(',')]

    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    print("Network Arch: %s", args.arch)
    print("CV split: %d", args.cv_split)
    print("Scales : %s", ' '.join(str(e) for e in scales))

    runner = RunEval(output_dir,
                     write_image=args.dump_images)
    net = get_net()

    image_names = os.listdir(args.input_path)
    output_names = [i.replace('.tif', '').replace('.png', '')
                    for i in image_names]
    images = [os.path.join(args.input_path, i) for i in image_names]

    # Run Inference!
    for idx in range(len(output_names)):
        img_names = output_names[idx]
        img = Image.open(images[idx]).convert('RGB')

        runner.inf(img, img_names, net, scales)
        print(idx, '/', len(output_names))
        if idx > 5 and args.test_mode:
            break


if __name__ == '__main__':
    main()
