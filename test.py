import network

import argparse
import jittor as jt

parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--arch', type=str,
                    default='network.pointflow_resnet_with_max_avg_pool.DeepR50_PF_maxavg_deeply')
args = parser.parse_args()

net = network.get_net(args, None)
net.eval()
inputs = jt.ones((1, 3, 224, 224))
main_loss_dic = net(inputs, gts=inputs)
