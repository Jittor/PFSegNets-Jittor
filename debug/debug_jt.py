import jittor as jt
from jittor import nn
import torch

class PSPModule(nn.Module):

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 5), norm_layer=nn.BatchNorm):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size, norm_layer) for size in sizes])

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv(features, out_features, 1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def execute(self, feats):
        for i in self.stages:
            tmp = feats
            for j in i:
                tmp = j(tmp)
                print(tmp.shape)
                print(tmp.max())


ppm = PSPModule(2048,256)
print(ppm)
ppm.load_parameters(jt.load('ppm.pth'))
feats = jt.array(torch.load('feats.pth',map_location=torch.device('cpu')).detach().numpy())
ppm(feats)
'''
tensor(5.7116, grad_fn=<MaxBackward1>)
tensor(1.7725, grad_fn=<MaxBackward1>)
tensor(4.1954, grad_fn=<MaxBackward1>)
tensor(5.7116, grad_fn=<MaxBackward1>)
tensor(1.5103, grad_fn=<MaxBackward1>)
tensor(4.4859, grad_fn=<MaxBackward1>)
tensor(5.7116, grad_fn=<MaxBackward1>)
tensor(1.6395, grad_fn=<MaxBackward1>)
tensor(4.5534, grad_fn=<MaxBackward1>)
tensor(5.7116, grad_fn=<MaxBackward1>)
tensor(1.8862, grad_fn=<MaxBackward1>)
tensor(4.8633, grad_fn=<MaxBackward1>)
'''