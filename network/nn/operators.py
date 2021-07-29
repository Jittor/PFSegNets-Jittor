import jittor as jt
from jittor import nn


class PSPModule(nn.Module):

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(nn.Conv((features + (len(sizes) * out_features)), out_features,
                                        1, padding=0, dilation=1, bias=False), norm_layer(out_features), nn.ReLU(), nn.Dropout(p=0.1))

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv(features, out_features, 1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def execute(self, feats):
        (h, w) = (feats.shape[2], feats.shape[3])
        priors = ([nn.interpolate(stage(feats), size=(
            h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats])
        bottle = self.bottleneck(jt.contrib.concat(priors, dim=1))
        return bottle
