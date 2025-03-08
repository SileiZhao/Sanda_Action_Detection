import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)

        return out


class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=10, dropout=0.5):
        super(SlowFast, self).__init__()

        # Fast Path for RGB
        self.fast_inplanes = 8
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU()
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        self.fast_res3 = self._make_layer_fast(block, 16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(block, 64, layers[3], stride=1, head_conv=3)

        # Lateral connections for RGB
        self.lateral_p1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32, 32 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False,
                                      padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64, 64 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False,
                                      padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128, 128 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False,
                                      padding=(2, 0, 0))

        # Slow Path for RGB
        self.slow_inplanes = 64 + 64 // 8 * 2 + 64 * 2
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU()
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res3 = self._make_layer_slow(block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(block, 512, layers[3], stride=1, head_conv=3)

        # Heatmap Path
        self.heatmap_inplanes = 64
        self.heatmap_conv1 = nn.Conv3d(17, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.heatmap_bn1 = nn.BatchNorm3d(64)
        self.heatmap_relu = nn.ReLU()
        self.heatmap_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.heatmap_res2 = self._make_layer_heatmap(block, 64, layers[0], head_conv=1)
        self.heatmap_res3 = self._make_layer_heatmap(block, 128, layers[1], stride=2, head_conv=1)
        self.heatmap_res4 = self._make_layer_heatmap(block, 256, layers[2], stride=2, head_conv=3)
        self.heatmap_res5 = self._make_layer_heatmap(block, 512, layers[3], stride=1, head_conv=3)

        # Lateral connections for Heatmap
        self.lateral_p1_heatmap = nn.Conv3d(64, 64 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False,
                                            padding=(2, 0, 0))
        self.lateral_res2_heatmap = nn.Conv3d(256, 256 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False,
                                              padding=(2, 0, 0))
        self.lateral_res3_heatmap = nn.Conv3d(512, 512 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False,
                                              padding=(2, 0, 0))
        self.lateral_res4_heatmap = nn.Conv3d(1024, 1024 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False,
                                              padding=(2, 0, 0))

        # Fusion layers
        self.fusion_fc = nn.Linear(512 * block.expansion + 2048 + 256, 1024)  # 512 (heatmap) + 2048 (slow) + 256 (fast)
        self.fc = nn.Linear(1024, class_num, bias=False)
        self.dp = nn.Dropout(dropout)

    def SlowPath(self, input, lateral_rgb, lateral_heatmap):  # (1, 3, 8, 256, 256)
        x = self.slow_conv1(input)  # (1, 64, 8, 128, 128)
        x = self.slow_bn1(x)  # (1, 64, 8, 128, 128)
        x = self.slow_relu(x)  # (1, 64, 8, 128, 128)
        x = self.slow_maxpool(x)  # (1, 64, 8, 64, 64)

        # 合并 RGB 和 Heatmap 的侧向连接
        x = torch.cat([x, lateral_rgb[0], lateral_heatmap[0]], dim=1)  # (1, 64 + 16 + 128, 8, 64, 64)
        x = self.slow_res2(x)  # (1, 256, 8, 64, 64)

        x = torch.cat([x, lateral_rgb[1], lateral_heatmap[1]], dim=1)  # (1, 256 + 64 + 512, 8, 64, 64)
        x = self.slow_res3(x)  # (1, 512, 8, 32, 32)

        x = torch.cat([x, lateral_rgb[2], lateral_heatmap[2]], dim=1)  # (1, 512 + 128 + 1024, 8, 32, 32)
        x = self.slow_res4(x)  # (1, 1024, 8, 16, 16)

        x = torch.cat([x, lateral_rgb[3], lateral_heatmap[3]], dim=1)  # (1, 1024 + 256 + 2048, 8, 16, 16)
        x = self.slow_res5(x)  # (1, 2048, 8, 16, 16)

        # print(f"slow: {x.shape}")
        # x = nn.AdaptiveAvgPool3d(1)(x)  # (1, 2048, 1, 1, 1)
        # x = x.view(-1, x.size(1))  # (1, 2048)
        return x

    def FastPath(self, input):  # (1, 3, 32, 256, 256)
        lateral = []
        x = self.fast_conv1(input)  # (1, 8, 32, 128, 128)
        x = self.fast_bn1(x)  # (1, 8, 32, 128, 128)
        x = self.fast_relu(x)  # (1, 8, 32, 128, 128)
        pool1 = self.fast_maxpool(x)  # (1, 8, 32, 64, 64)
        lateral_p = self.lateral_p1(pool1)  # (1, 16, 8, 64, 64)
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)  # (1, 32, 32, 64, 64)
        lateral_res2 = self.lateral_res2(res2)  # (1, 64, 8, 64, 64)
        lateral.append(lateral_res2)

        res3 = self.fast_res3(res2)  # (1, 64, 16, 32, 32)
        lateral_res3 = self.lateral_res3(res3)  # (1, 128, 8, 32, 32)
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)  # (1, 128, 32, 16, 16)
        lateral_res4 = self.lateral_res4(res4)  # (1, 256, 8, 16, 16)
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)  # (1, 256, 32, 16, 16)
        # print(f"fast: {res5.shape}")
        # x = nn.AdaptiveAvgPool3d(1)(res5)  # (1, 256, 1, 1, 1)
        # x = x.view(-1, x.size(1))  # (1, 256)

        return res5, lateral  # lateral: [(1, 16, 8, 64, 64), (1, 64, 8, 64, 64), (1, 128, 8, 32, 32), (1, 256, 8, 16, 16)]

    def HeatmapPath(self, heatmap):
        # heatmap shape: (batch_size, frame_num, num_joints, height, width)
        batch_size, frame_num, num_joints, height, width = heatmap.shape

        # Reshape to (batch_size, num_joints, frame_num, height, width)
        heatmap = heatmap.permute(0, 2, 1, 3, 4)

        # Heatmap feature extraction
        lateral = []
        x = self.heatmap_conv1(heatmap)  # (batch_size, 64, frame_num, height//2, width//2)
        x = self.heatmap_bn1(x)
        x = self.heatmap_relu(x)
        pool1 = self.heatmap_maxpool(x)  # (batch_size, 64, frame_num, height//4, width//4)
        lateral_p = self.lateral_p1_heatmap(pool1)  # (batch_size, 128, frame_num//4, height//4, width//4)
        lateral.append(lateral_p)

        res2 = self.heatmap_res2(pool1)  # (batch_size, 256, frame_num, height//4, width//4)
        lateral_res2 = self.lateral_res2_heatmap(res2)  # (batch_size, 512, frame_num//4, height//4, width//4)
        lateral.append(lateral_res2)

        res3 = self.heatmap_res3(res2)  # (batch_size, 512, frame_num//2, height//8, width//8)
        lateral_res3 = self.lateral_res3_heatmap(res3)  # (batch_size, 1024, frame_num//4, height//8, width//8)
        lateral.append(lateral_res3)

        res4 = self.heatmap_res4(res3)  # (batch_size, 1024, frame_num, height//16, width//16)
        lateral_res4 = self.lateral_res4_heatmap(res4)  # (batch_size, 2048, frame_num//4, height//16, width//16)
        lateral.append(lateral_res4)

        res5 = self.heatmap_res5(res4)  # (batch_size, 2048, frame_num, height//16, width//16)
        # print(f"Heatmap: {res5.shape}")

        # Global average pooling
        # x = nn.AdaptiveAvgPool3d(1)(res5)  # (batch_size, 2048, 1, 1, 1)
        # x = x.view(batch_size, -1)  # (batch_size, 2048)

        return res5, lateral  # 返回特征和侧向连接

    def forward(self, input, pose):
        # RGB paths
        fast, lateral_rgb = self.FastPath(input[:, :, ::1, :, :])

        # Heatmap path
        heatmap = self.generate_heatmap(pose)  # Generate heatmap from pose
        heatmap_feat, lateral_heatmap = self.HeatmapPath(heatmap)

        # Slow path with lateral connections from RGB and Heatmap
        slow = self.SlowPath(input[:, :, ::4, :, :], lateral_rgb, lateral_heatmap)

        # Fusion
        # x = torch.cat([slow, fast, heatmap_feat], dim=1)  # (1, 2048 + 256 + 512 * expansion)
        # x = self.fusion_fc(x)  # (1, 1024)
        # x = self.dp(x)
        # x = self.fc(x)  # (1, num_class)
        return fast, heatmap_feat, slow

    def generate_heatmap(self, pose):
        # pose shape: (batch_size, frame_num, N, 17, 3)
        batch_size, frame_num, N, num_joints, _ = pose.shape
        height, width = 256, 256  # Heatmap resolution (same as input RGB resolution)

        # Initialize heatmap
        heatmap = torch.zeros(batch_size, frame_num, num_joints, height, width).to(pose.device)

        # Generate Gaussian heatmap for each joint
        for b in range(batch_size):
            for f in range(frame_num):
                for n in range(N):
                    for j in range(num_joints):
                        x, y, score = pose[b, f, n, j]
                        if score > 0:  # Only generate heatmap for valid joints
                            heatmap[b, f, j] += self.gaussian_heatmap(x, y, height, width)

        # Average over N people (keep num_joints dimension)
        heatmap = torch.mean(heatmap, dim=2, keepdim=True)  # (batch_size, frame_num, 1, height, width)
        heatmap = heatmap.expand(-1, -1, num_joints, -1, -1)  # (batch_size, frame_num, num_joints, height, width)

        return heatmap

    def gaussian_heatmap(self, x, y, height, width, sigma=2):
        # Create grid
        x_grid = torch.arange(width).float().to(x.device)
        y_grid = torch.arange(height).float().to(y.device)
        y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')

        # Generate Gaussian distribution
        heatmap = torch.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
        return heatmap

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))

        self.slow_inplanes = planes * block.expansion + planes * block.expansion // 8 * 2 + planes * block.expansion * 2
        return nn.Sequential(*layers)

    def _make_layer_heatmap(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.heatmap_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.heatmap_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.heatmap_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.heatmap_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.heatmap_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = SlowFast(class_num=2)
    rgb = torch.autograd.Variable(torch.randn(1, 3, 32, 256, 256))
    pose = torch.autograd.Variable(torch.randn(1, 32, 2, 17, 3))
    out = model(rgb, pose)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
