import jittor as jt
from jittor import nn


def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if (point_coords.ndim == 3):
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = nn.grid_sample(input, ((2.0 * point_coords) - 1.0), **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    (R, _, H, W) = uncertainty_map.shape
    h_step = (1.0 / float(H))
    w_step = (1.0 / float(W))
    num_points = min((H * W), num_points)
    point_indices = jt.topk(uncertainty_map.view(
        (R, (H * W))), k=num_points, dim=1)[1]
    point_coords = jt.zeros((R, num_points, 2))
    point_coords[:, :, 0] = (
        (w_step / 2.0) + ((point_indices % W) * w_step))
    point_coords[:, :, 1] = (
        (h_step / 2.0) + ((point_indices // W) * h_step))
    return (point_indices, point_coords)


class PointMatcher(nn.Module):

    def __init__(self, dim, kernel_size=3):
        super(PointMatcher, self).__init__()
        self.match_conv = nn.Conv((dim * 2), 1, kernel_size, padding=1)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        (x_high, x_low) = x
        x_low = nn.upsample(
            x_low, size=x_high.shape[2:], mode='bilinear', align_corners=True)
        certainty = self.match_conv(jt.contrib.concat([x_high, x_low], dim=1))
        return self.sigmoid(certainty)


class PointFlowModuleWithMaxAvgpool(nn.Module):

    def __init__(self, in_planes, dim=64, maxpool_size=8, avgpool_size=8, matcher_kernel_size=3, edge_points=64):
        super(PointFlowModuleWithMaxAvgpool, self).__init__()
        self.dim = dim
        self.point_matcher = PointMatcher(dim, matcher_kernel_size)
        self.down_h = nn.Conv(in_planes, dim, 1)
        self.down_l = nn.Conv(in_planes, dim, 1)
        self.softmax = nn.Softmax(dim=(- 1))
        self.maxpool_size = maxpool_size
        self.avgpool_size = avgpool_size
        self.edge_points = edge_points
        # self.max_pool = nn.AdaptiveMaxPool2d(
        #     (maxpool_size, maxpool_size))
        self.max_pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.max_pool2 = nn.MaxPool2d(4, 4, return_indices=True)
        self.max_pool3 = nn.MaxPool2d(8, 8, return_indices=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))
        self.edge_final = nn.Sequential(nn.Conv(in_planes, in_planes, 3, padding=1, bias=False), nn.BatchNorm(
            in_planes), nn.ReLU(), nn.Conv(in_planes, 1, 3, padding=1, bias=False))

    def execute(self, x):
        (x_high, x_low) = x
        stride_ratio = (x_low.shape[2] / x_high.shape[2])
        x_high_embed = self.down_h(x_high)
        x_low_embed = self.down_l(x_low)
        (N, C, H, W) = x_low.shape
        (N_h, C_h, H_h, W_h) = x_high.shape
        certainty_map = self.point_matcher([x_high_embed, x_low_embed])
        avgpool_grid = self.avg_pool(certainty_map)
        (_, _, map_h, map_w) = certainty_map.shape
        avgpool_grid = nn.interpolate(avgpool_grid, size=(
            map_h, map_w), mode='bilinear', align_corners=True)
        x_high_edge = (x_high - (x_high * avgpool_grid))
        edge_pred = self.edge_final(x_high_edge)
        (point_indices, point_coords) = get_uncertain_point_coords_on_grid(
            edge_pred, num_points=self.edge_points)
        sample_x = ((point_indices % W_h) * stride_ratio)
        sample_y = ((point_indices // W_h) * stride_ratio)
        low_edge_indices = (sample_x + (sample_y * W))
        low_edge_indices = low_edge_indices.unsqueeze(
            1).expand((- 1), C, (- 1)).long()
        high_edge_feat = point_sample(x_high, point_coords)
        low_edge_feat = point_sample(x_low, point_coords)
        affinity_edge = nn.bmm(high_edge_feat.transpose(
            0, 2, 1), low_edge_feat).transpose(0, 2, 1)
        affinity = self.softmax(affinity_edge)
        high_edge_feat = nn.bmm(
            affinity, high_edge_feat.transpose(0, 2, 1)).transpose(0, 2, 1)
        fusion_edge_feat = (high_edge_feat + low_edge_feat)
        maxpool_grid = None
        maxpool_indices = None
        if certainty_map.shape[2] == 28:
            (maxpool_grid, maxpool_indices) = self.max_pool1(certainty_map)
        elif certainty_map.shape[2] == 56:
            (maxpool_grid, maxpool_indices) = self.max_pool2(certainty_map)
        elif certainty_map.shape[2] == 112:
            (maxpool_grid, maxpool_indices) = self.max_pool3(certainty_map)
        maxpool_indices = maxpool_indices.expand((- 1), C, (- 1), (- 1))
        maxpool_grid = nn.interpolate(maxpool_grid, size=(
            map_h, map_w), mode='bilinear', align_corners=True)
        x_indices = ((maxpool_indices % W_h) * stride_ratio)
        y_indices = ((maxpool_indices // W_h) * stride_ratio)
        low_indices = (x_indices + (y_indices * W))
        low_indices = low_indices.long()
        x_high = (x_high + (maxpool_grid * x_high))
        flattened_high = x_high.flatten(start_dim=2)
        high_features = jt.gather(flattened_high,
                                  dim=2, index=maxpool_indices.flatten(start_dim=2)).view_as(maxpool_indices)
        flattened_low = x_low.flatten(start_dim=2)
        low_features = jt.gather(flattened_low,
                                 dim=2, index=low_indices.flatten(start_dim=2)).view_as(low_indices)
        (feat_n, feat_c, feat_h, feat_w) = high_features.shape
        high_features = high_features.view((feat_n, (- 1), (feat_h * feat_w)))
        low_features = low_features.view((feat_n, (- 1), (feat_h * feat_w)))
        affinity = nn.bmm(high_features.transpose(
            0, 2, 1), low_features).transpose(0, 2, 1)
        affinity = self.softmax(affinity)
        high_features = nn.bmm(
            affinity, high_features.transpose(0, 2, 1)).transpose(0, 2, 1)
        fusion_feature = (high_features + low_features)
        (mp_b, mp_c, mp_h, mp_w) = low_indices.shape
        low_indices = low_indices.view((mp_b, mp_c, (- 1)))
        final_features = jt.scatter(jt.reshape(
            x_low, (N, C, H*W)), 2, low_edge_indices, fusion_edge_feat)
        final_features = jt.scatter(final_features,
                                    2, low_indices, fusion_feature).view(N, C, H, W)
        return (final_features, edge_pred)
