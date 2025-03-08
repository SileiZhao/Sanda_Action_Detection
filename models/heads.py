import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align as torchvision_roi_align


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale=1.0, sampling_ratio=0, aligned=True):
        """
        Args:
            output_size (tuple[int]): 输出的空间大小 (height, width)。
            spatial_scale (float): 输入特征图与原始图像的比例（例如，特征图是原始图像的 1/16，则 scale=1/16）。
            sampling_ratio (int): 每个 ROI 中用于插值的采样点数量。如果为 0，则自动计算。
            aligned (bool): 是否对齐插值点。如果为 True，插值点会更精确地对齐像素中心。
        """
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois):
        """
        Args:
            input (Tensor): 输入特征图，形状为 (batch_size, channels, height, width)。
            rois (Tensor): ROI 框，形状为 (num_rois, 5)，每行格式为 (batch_index, x1, y1, x2, y2)。
        Returns:
            output (Tensor): ROIAlign 后的特征，形状为 (num_rois, channels, output_height, output_width)。
        """
        # 调用 PyTorch 的 roi_align 函数
        output = torchvision_roi_align(
            input,
            rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )

        return output


class ResNetRoIHead(nn.Module):
    def __init__(
            self,
            dim_in,  # 例如 [256, 2048, 2048]，表示三个路径的通道数
            num_classes,
            pool_size,  # 例如 [[1, 7, 7], [1, 7, 7], [1, 7, 7]]
            resolution,  # 例如 [[7, 7], [7, 7], [7, 7]]
            scale_factor,  # 例如 [16, 16, 16]
            dropout_rate=0.0,
            act_func="softmax",
            aligned=True,
            detach_final_fc=False,
    ):
        super(ResNetRoIHead, self).__init__()
        assert (
                len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.detach_final_fc = detach_final_fc

        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d([pool_size[pathway][0], 1, 1], stride=1)
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # 激活函数
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
                len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])  # 时间池化
            assert out.shape[2] == 1, f"时间池化后时间维度应为 1，但得到 {out.shape[2]}"
            out = torch.squeeze(out, 2)  # 去除时间维度

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)  # ROIAlign, 输出形状为 (batch_size * clip_length, channel, h, w)

            # 将 (batch_size * clip_length, channel, h, w) 转换为 (batch_size, clip_length * channel, h, w)
            batch_size = inputs[0].size(0)  # 原始 batch_size
            clip_length = inputs[0].size(2)  # 原始 clip_length
            out = out.view(batch_size, clip_length * out.size(1), out.size(2),
                           out.size(3))  # (batch_size, clip_length * channel, h, w)

            # 全局平均池化
            out = self.global_avg_pool(out)  # (batch_size, clip_length * channel, 1, 1)
            out = out.view(out.size(0), -1)  # (batch_size, clip_length * channel)

            # 将 (batch_size, clip_length * channel) 转换为 (batch_size, channel)
            out = out.view(batch_size, clip_length, -1)  # (batch_size, clip_length, channel)
            out = torch.mean(out, dim=1)  # (batch_size, channel)

            pool_out.append(out)

        # 拼接所有路径的特征
        x = torch.cat(pool_out, 1)  # (batch_size, sum(dim_in))
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        # 全连接层
        x = self.projection(x)  # (batch_size, num_classes)
        x = self.act(x)
        return x


if __name__ == '__main__':
    # 输入特征维度
    fast = torch.randn(1, 256, 32, 16, 16)  # (batch_size, channels, temporal_dim, height, width)
    heatmap = torch.randn(1, 2048, 32, 16, 16)
    slow = torch.randn(1, 2048, 8, 16, 16)

    # ROI 框 (num_rois=2, 每行格式为 (batch_index, x1, y1, x2, y2))
    rois = torch.tensor([
        [0, 0, 0, 15, 15],  # 第一个样本的 ROI
        [0, 5, 5, 10, 10],  # 第一个样本的另一个 ROI
        # [2, 1, 1, 5, 5],
        # [2, 0, 0, 12, 12],
        # [3, 0, 0, 10, 10]
    ], dtype=torch.float32)

    # ResNetRoIHead 参数
    dim_in = [256, 2048, 2048]  # 三个路径的通道数
    num_classes = 2
    pool_size = [[32, 2, 2], [32, 2, 2], [8, 2, 2]]  # 时间池化大小
    resolution = [[8, 8], [8, 8], [8, 8]]  # ROIAlign 输出大小
    scale_factor = [16, 16, 16]  # 空间缩放因子

    # 初始化 ResNetRoIHead
    roi_head = ResNetRoIHead(
        dim_in=dim_in,
        num_classes=num_classes,
        pool_size=pool_size,
        resolution=resolution,
        scale_factor=scale_factor,
        dropout_rate=0.5,
        act_func="softmax",
        aligned=True,
    )

    # 前向传播
    inputs = [fast, heatmap, slow]
    outputs = roi_head(inputs, rois)
    print("ResNetRoIHead 输出形状:", outputs.shape)  # 应为 (num_rois, num_classes)
