import torch
import torch.nn as nn

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 5), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        for i in self.stages:
            tmp = feats
            for j in i:
                tmp = j(tmp)
                print(tmp.max())

ppm = PSPModule(2048,256)
print(ppm)
ppm.load_state_dict(torch.load('ppm.pth'))
feats = torch.load('feats.pth',map_location=torch.device('cpu'))
ppm(feats)