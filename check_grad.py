import os
import pickle as pk
import numpy as np
import torch

def check_diff(w1_, w2_):
    pt = torch.load(w1_,map_location=torch.device('cpu'))
    w1 = {}
    for p in pt.keys():
        numpy_para = pt[p].cpu().numpy()
        w1[p.replace('module.','')] = numpy_para
    
    w2 = pk.load(open(w2_, "rb"))
    keys = list(w1.keys())
    keys.sort()
    for k in keys:
        if 'num_batches_tracked' in k:
            continue
        v1 = w1[k]
        v2 = w2[k]
        abs_err = (np.abs(v1 - v2)).mean()
        rel_err = (np.abs(v1 - v2) / (np.maximum(np.abs(v1), np.abs(v2)) + 1e-12)).mean()
        our_err = (np.abs(v1 - v2) / (np.abs(v1) + 1e-12)).mean()
        print(f'{k:45}{abs_err:10.5f}{rel_err:10.5f}{our_err:20.5f}')

def main():
    path1 = '/home/gmh/project/yizhang/PFSegNets/net_torch.pth'
    path2 = '/home/gmh/project/yizhang/PFSegNets-Jittor/net_jt.pkl'
    check_diff(path1, path2)
main()