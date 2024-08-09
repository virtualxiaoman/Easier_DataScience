import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import os
import pandas as pd
from easier_excel.read_data import show_images


# 从（左上，右下）转换到（中间，宽度，高度）
def box_corner_to_center(boxes):
    """
    从（左上，右下）转换到（中间，宽度，高度）
    :param boxes: shape: (n, 4)，n是边界框的个数，4是左上和右下的坐标
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), dim=-1)  # dim=-1表示最后一个维度，即坐标维度
    return boxes


# 从（中间，宽度，高度）转换到（左上，右下）
def box_center_to_corner(boxes):
    """
    从（中间，宽度，高度）转换到（左上，右下）
    注意：显示器图像的坐标系是左上角为原点，x方向向右为正，y方向向下为正
    :param boxes: shape: (n, 4)
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes


# 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式： ((左上x,左上y),宽,高)
def bbox_to_rect(bbox, color):
    """将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式： ((左上x,左上y),宽,高)"""
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1], fill=False,
                         edgecolor=color, linewidth=2)


# 生成以每个像素为中心具有不同形状的锚框
def multibox_prior(data, sizes, ratios):
    """
    生成以每个像素为中心具有不同形状的锚框
    :param data: 输入数据，形状为(批量大小, 通道数, 高, 宽)
    :param sizes: 缩放比scale，长度为n
    :param ratios: 宽高比，长度为m
    :return: 整个输入图像的锚框，形状为(1, 高*宽*锚框数, 4)，其中锚框数(以同一像素为中心的锚框的数量)=n + m − 1
    """
    in_height, in_width = data.shape[-2:]
    device, n, m = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (n + m - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = (torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:])))
         * in_height / in_width)  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


# 显示所有边界框
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


# 对于两个边界框，它们的杰卡德系数通常称为交并比（intersection over union，IoU），即两个边界框相交面积与相并面积之比
def box_iou(boxes1, boxes2):
    """
    计算两个锚框或边界框列表中成对的交并比(两个边界框相交面积与相并面积之比)
     .. math::
        J({A},{B})=\frac{|{A}\cap{B}|}{|{A}\cup{B}|} \in [0, 1]
    :param boxes1: 第一个锚框或边界框列表，shape: (boxes1的数量, 4)
    :param boxes2: 第二个锚框或边界框列表，shape: (boxes2的数量, 4)
    :return: 交并比，shape: (boxes1的数量, boxes2的数量)
    """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    # boxes1：(boxes1的数量,4), boxes2：(boxes2的数量,4), areas1：(boxes1的数量,), areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts, inter_lowerrights,inters的形状: (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areas, union_areas的形状: (boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


# 为每个锚框分配真实边界框标签
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """
    将最接近的真实边界框分配给锚框。
    给定图像，假设锚框是$A_1, A_2, \ldots, A_{n_a}$，真实边界框是$B_1, B_2, \ldots, B_{n_b}$，其中$n_a \geq n_b$。
    矩阵${X} \in \mathbb{R}^{n_a \times n_b}$，$x_{ij}$是锚框$A_i$和真实边界框$B_j$的IoU。
    1. 找到矩形$X$中最大值的索引$(i, j)$，对锚框$A_i$，其最接近的真实边界框是$B_j$。然后忽略$i$行和$j$列。
    2. 在剩余元素中重复1的过程。
    3. 对于剩余的$n_a - n_b$个锚框，依次检测其IoU是否大于阈值，如果大于则分配为对应的真实边界框，否则分配为背景。
    :param ground_truth: 真实边界框，shape: (真实边界框的数量, 4)
    :param anchors: 锚框，shape: (锚框的数量, 4)
    :param device: torch.device
    :param iou_threshold: 重叠IoU阈值
    :return: 锚框的标签，shape: (锚框的数量,)
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


# 对锚框偏移量的转换
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """
    对锚框偏移量的转换
    给定框$A$和$B$，中心坐标分别为$(x_a, y_a)$和$(x_b, y_b)$，宽度/高度分别为$w_a$和$w_b$. $h_a$和$h_b$，$A$的偏移量为：
        $$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
        \frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
        \frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
        \frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right)$$
    常量的默认值为 $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$ ， $\sigma_w=\sigma_h=0.2$。
    :param anchors: 锚框，shape: (锚框的数量, 4)
    :param assigned_bb: 真实边界框，shape: (锚框的数量, 4)
    :param eps: 一个极小值，防止被零除
    :return: 偏移量，shape: (锚框的数量, 4)
    """
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset


# 使用真实边界框标记锚框
def multibox_target(anchors, labels):
    """
    使用真实边界框标记锚框
    :param anchors: 锚框，shape: (1, 锚框的数量, 4)
    :param labels: 真实边界框，shape: (批量大小, 真实边界框的数量, 5)
    :return: bbox_offset, bbox_mask, class_labels
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return bbox_offset, bbox_mask, class_labels


# 将锚框和偏移量预测作为输入，并应用逆偏移变换来返回预测的边界框坐标
def offset_inverse(anchors, offset_preds):
    """
    根据带有预测偏移量的锚框来预测边界框
    :param anchors: 锚框，shape: (锚框的数量, 4)
    :param offset_preds: 预测的偏移量，shape: (批量大小, 锚框的数量*4)
    :return: 预测的边界框，shape: (批量大小, 锚框的数量, 4)
    """
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


# 使用非极大值抑制（non‐maximum suppression，NMS）合并属于同一目标的类似的预测边界框
def nms(boxes, scores, iou_threshold):
    """
    对预测边界框的置信度进行排序
    :param boxes: 预测边界框，shape: (预测边界框的数量, 4)
    :param scores: 预测边界框的置信度
    :param iou_threshold: 交并比阈值
    :return: 非极大值抑制后保留的锚框的指标 list，shape: (保留的锚框的数量,)
    """
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1:
            break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


# 预测边界框
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)


# 读取香蕉检测数据集中的图像和标签
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = "../data/banana-detection"
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


# 自定义数据集
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)


# 加载香蕉检测数据集
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True), batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False), batch_size)
    return train_iter, val_iter


# 类别预测层
def cls_predictor(num_inputs, num_anchors, num_classes):
    """
    创建一个类别预测层
    :param num_inputs: 输入的通道数
    :param num_anchors: 锚框的数量
    :param num_classes: 类别的数量
    :return: 类别预测层
    """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


# 边界框预测层
def bbox_predictor(num_inputs, num_anchors):
    """每个锚框预测4个偏移量，而不是q+1个类别。"""
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


# 将预测结果转成二维的（批量大小，高×宽×通道数）的格式
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


# 1. 展示图像
img = plt.imread('../data/catdog.jpg')  # shape:(561, 728, 3)
h, w = img.shape[:2]  # 561 728
plt.imshow(img)
plt.show()

# 2. 展示边界框
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]  # bbox：边界框
boxes = torch.tensor((dog_bbox, cat_bbox))  # shape: (2, 4)
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)  # 验证转换函数是否正确
fig = plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))  # add_patch：添加图形
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
plt.show()

# 3. 生成锚框
# X是img，permute(2, 0, 1)将通道数调整到第二个维度，unsqueeze(0)添加一维，float()转换为浮点数
X = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()  # shape: (1, 3, 561, 728) = (批量大小, 通道数, 高, 宽)
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])  # shape: (1, 2042040, 4)，2042040=561*728*5
print(Y.shape)
boxes = Y.reshape(h, w, 5, 4)  # 5是每个像素的锚框数，4是左上和右下的坐标
print(boxes[250, 250, 0, :])  # tensor([0.0551, 0.0715, 0.6331, 0.8215])
bbox_scale = torch.tensor((w, h, w, h))  # tensor([728, 561, 728, 561])
fig = plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,  # 乘以bbox_scale将坐标转换为图像坐标
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])
plt.show()


# 4. 为每个锚框分配真实边界框标签
# 真实边界框。第一个元素是类别（0代表狗，1代表猫），其余四个元素是左上角和右下角的(x, y)轴坐标（范围介于0和1之间）
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],  [1, 0.55, 0.2, 0.9, 0.88]])
# 五个锚框。用左上角和右下角的坐标进行标记
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4], [0.63, 0.05, 0.88, 0.98],
                        [0.66, 0.45, 0.8, 0.8], [0.57, 0.3, 0.92, 0.9]])
fig = plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
plt.show()
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
# 类别标签 tensor([[0, 1, 2, 0, 2]])，锚框0、1、2分别为狗、猫、背景
print(labels[2])
# 掩码 形状为（批量大小，锚框数*4），用于过滤掉负类偏移量
# tensor([[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.]])
print(labels[1])
torch.set_printoptions(precision=2, sci_mode=False)  # 设置2位小数，不要使用科学计数法
# 偏移量 形状为（批量大小，锚框数*4）。其中负类锚框的偏移量被标记为0
# tensor([[    -0.00,     -0.00,     -0.00,     -0.00,      1.40,     10.00,
#               2.59,      7.18,     -1.20,      0.27,      1.68,     -1.57,
#              -0.00,     -0.00,     -0.00,     -0.00,     -0.57,     -1.00,
#               0.00,      0.63]])
print(labels[0])


# 5. 预测边界框
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95], [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
fig = plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
plt.show()
output = multibox_detection(cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0), nms_threshold=0.5)
print(output)
fig = plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
plt.show()


# 6. 读取banana数据集
batch_size, edge_size = 32, 256  # 批量大小, 图像的边缘长度
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
# 图像小批量的形状为（批量大小、通道数、高度、宽度），标签的小批量的形状为（批量大小，m，5），m是边界框可能出现的最大数量。
print(batch[0].shape, batch[1].shape)  # (torch.Size([32, 3, 256, 256]), torch.Size([32, 1, 5]))
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = show_images(imgs, 2, 5)
for ax, label in zip(axes, batch[1][0:10]):
    show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
plt.show()

# 7. TinySSD模型
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)
print('output anchors:', anchors.shape)  # [1, 5444, 4]
print('output class preds:', cls_preds.shape)  # [32, 5444, 2]
print('output bbox preds:', bbox_preds.shape)  # [32, 21776]

# 8. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
train_iter, _ = load_data_bananas(batch_size)
net = TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


num_epochs = 10
net = net.to(device)
for epoch in range(num_epochs):
    cls_correct_sum, cls_total_sum = 0, 0
    bbox_mae_sum, bbox_total_sum = 0.0, 0
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    net.train()
    for features, target in train_iter:
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        # 计算分类正确的个数和总数
        cls_correct_sum += (cls_preds.argmax(dim=2) == cls_labels).sum().item()
        cls_total_sum += cls_labels.numel()
        # 计算边界框预测的平均绝对误差
        bbox_mae_sum += (torch.abs(bbox_preds - bbox_labels) * bbox_masks).sum().item()
        bbox_total_sum += bbox_labels.numel()
    cls_err = 1 - cls_correct_sum / cls_total_sum  #
    bbox_mae = bbox_mae_sum / bbox_total_sum
    print(f'epoch {epoch + 1}, class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')


X = torchvision.io.read_image('../data/banana-detection/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()


def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


def display(img, output, threshold):
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')


output = predict(X)
display(img, output.cpu(), threshold=0.9)
plt.show()
