import argparse
import os
import cv2
import numpy as np
import torch
from PIL.ImImagePlugin import number

from utils_1.utils import generate_bbox, py_nms, convert_to_square
from utils_1.utils import pad, calibrate_box, processed_image
from utils_1.data import extract_combined_elbp_dlbp, preprocess_image, adjust_contrast
# 创建参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='...', help='PNet、RNet、ONet三个模型文件存在的文件夹路径')
parser.add_argument('--image_paths', type=str, nargs='+', default=[], help='需要预测图像的路径列表')
args = parser.parse_args()

# 如果没有通过命令行参数提供图像路径，尝试从指定的文件中读取
if not args.image_paths:
    file_path = "..."  # 存储文件名和数据的txt文件路径
    image_paths = []   # 测试集，每张图片地址
    four_xy = []   # 实际框（4个坐标）
    ten_xy = []   # 关键点（10个）

    with open(file_path, "r") as file:
        for line in file:
            part = line.strip().split()
            image_path ="..." + part[0].replace('\\',"/")
            image_paths.append(image_path)

            four = list(map(int, part[1:5]))  # 四个坐标
            four_xy.append(four)

            ten = list(map(float, part[5:]))  # 其余数据
            ten_xy.append(ten)

    args.image_paths = image_paths  # 将读取的路径赋值给 args.image_paths

# 打印结果以确认
print("已经载入数据")

device = torch.device('cpu')
# device = torch.device("cuda")


# 获取P模型
pnet = torch.jit.load(os.path.join("...", 'PNet.pth'))
pnet.to(device)
softmax_p = torch.nn.Softmax(dim=0)
pnet.eval()

# 获取R模型
rnet = torch.jit.load(os.path.join("...", 'RNet.pth'))
rnet.to(device)
softmax_r = torch.nn.Softmax(dim=-1)
rnet.eval()

# 获取O模型
onet = torch.jit.load(os.path.join("...", 'ONet.pth'))
onet.to(device)
softmax_o = torch.nn.Softmax(dim=-1)
onet.eval()


# 使用PNet模型预测
def predict_pnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    # 执行预测
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用RNet模型预测
def predict_rnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用ONet模型预测
def predict_onet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, landmark_pred = onet(infer_data)
    cls_prob = softmax_o(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()


# 获取PNet网络输出结果
def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    """
    net_size = 12
    # 人脸和输入图像的比率
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    # 图像金字塔
    while min(current_height, current_width) > net_size:
        # 类别和box
        cls_cls_map, reg = predict_pnet(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor  # 继续缩小图像做金字塔
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape

        if boxes.size == 0:
            continue
        # 非极大值抑制留下重复低的box
        keep = py_nms(boxes[:, :5], 0.5, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    # 将金字塔之后的box也进行非极大值抑制
    keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
    all_boxes = all_boxes[keep]
    # box的长宽
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    # 对应原图的box坐标和分数
    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])
    boxes_c = boxes_c.T

    return boxes_c


# 获取RNet网络输出结果
def detect_rnet(im, dets, thresh):
    """通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
    """
    h, w, c = im.shape

    # 将pnet的box变成包含它的正方形，可以避免信息损失
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    # 调整超出图像的box
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 4, 24, 24), dtype=np.float32)
    for i in range(int(num_boxes)):
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 4), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        except:
            continue
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None

    keep = py_nms(boxes, 0.4, mode='Union')
    boxes = boxes[keep]
    # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes_c


# 获取ONet模型预测结果
def detect_onet(im, dets, thresh):
    """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    num_boxes = dets.shape[0]
    cropped_ims = np.zeros((num_boxes, 4, 48, 48), dtype=np.float32)
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 4), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) / 128
        cropped_ims[i, :, :, :] = img
    cls_scores, reg, landmark = predict_onet(cropped_ims)

    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        landmark = landmark[keep_inds]
    else:
        return None, None

    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
    landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
    boxes_c = calibrate_box(boxes, reg)

    keep = py_nms(boxes_c, 0.6, mode='Minimum')
    boxes_c = boxes_c[keep]
    landmark = landmark[keep]
    return boxes_c, landmark


# 预测图片
def infer_image(image_paths):
    im = cv2.imread(image_paths, cv2.IMREAD_UNCHANGED)
    if im is not None and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)  # 转换为RGBA
        im[..., 3] = extract_combined_elbp_dlbp(im)
        # 调用第一个模型预测
    boxes_c = detect_pnet(im, 20, 0.79, 0.9)
    if boxes_c is None:
        return [[0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # 调用第二个模型预测
    boxes_c = detect_rnet(im, boxes_c, 0.6)
    if boxes_c is None:
        return [[0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # 调用第三个模型预测
    boxes_c, landmark = detect_onet(im, boxes_c, 0.7)
    if boxes_c is None:
        return [[0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    return boxes_c, landmark



def compute_iou(true_box, pred_box):
    """
    计算真实框和预测框的 IoU 值
    参数：
        true_box: [y1, x1, x2, y2] 格式的真实框
        pred_box: [y1, x1, x2, y2, score] 格式的预测框，取前 4 个
    返回：
        iou: 交并比值 (0-1)
    """
    # 提取坐标
    x1_true, y1_true, wide, high = true_box
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box[:4]
    x2_true = x1_true + wide
    y2_true = y1_true + high
    # 计算交集
    x1_inter = max(x1_true, x1_pred)
    y1_inter = max(y1_true, y1_pred)
    x2_inter = min(x2_true, x2_pred)
    y2_inter = min(y2_true, y2_pred)

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # 计算面积
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union_area = true_area + pred_area - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def compute_keypoint_accuracy(true_keypoints, pred_keypoints, threshold=20.0):
    """
    计算关键点的准确率，基于欧氏距离误差
    """
    num_keypoints = len(true_keypoints) // 2
    correct_count = 0
    for i in range(num_keypoints):
        true_x = true_keypoints[2 * i]
        true_y = true_keypoints[2 * i + 1]
        pred_x = pred_keypoints[2 * i]
        pred_y = pred_keypoints[2 * i + 1]

        distance = np.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)

        if distance <= threshold:
            correct_count += 1

    accuracy = correct_count / num_keypoints
    return accuracy


def evaluate_batch(true_boxes, pred_boxes, true_keypoints, pred_keypoints, iou_threshold=0.5, kp_threshold=20.0):
    """
    批量计算 IoU 和关键点准确率，并返回总体的框和关键点的准确度
    以及统计框精准度和关键点精准度大于 0.8 的百分比
    """
    id = 0
    id_list = []
    boxes_c_i = []
    box_matches = []
    keypoint_matches = []
    iou_all_scores = []  # 修改：变量名从 iou_all 改为 iou_all_scores
    iou_matched_scores = []  # 修改：变量名从 iou_part 改为 iou_matched_scores

    # 计算框的匹配情况
    for i in range(len(true_boxes)):
        matched = False
        max_kp_acc = 0
        max_c = 0
        true_box = true_boxes[i]
        print(true_box)
        pred_box = pred_boxes[i]
        print(pred_box)
        true_kp = true_keypoints[i]
        print(true_kp)
        pred_kp = pred_keypoints[i]
        print(pred_kp)

        if len(pred_box) > 1:
            for j in range(len(pred_box)):
                if pred_box[j][-1] > max_c:
                    max_c = pred_box[j][-1]
                    max_iou = compute_iou(true_box[:4], pred_box[j][:4])
                    mac_acc = compute_keypoint_accuracy(true_kp, pred_kp[j], threshold=kp_threshold)
        else:
            max_c = pred_box[0][-1]
            max_iou = compute_iou(true_box[:4], pred_box[0][:4])
            mac_acc = compute_keypoint_accuracy(true_kp, pred_kp[0], threshold=kp_threshold)
        print(max_iou)
        print("\n")
        id += 1
        if max_iou >= iou_threshold:
            matched = True
            iou_matched_scores.append(max_iou)
        else:
            id_list.append(id)
        boxes_c_i.append(max_c)
        iou_all_scores.append(max_iou)
        box_matches.append(matched)
        if mac_acc >= 0.6:  # 设定阈值
            matched = True
        keypoint_matches.append(matched)

    # 计算框的整体准确率
    box_accuracy = np.mean(box_matches)

    # 计算关键点的整体准确率
    keypoint_accuracy = np.mean(keypoint_matches)

    # 计算框和关键点同时匹配的准确率
    both_acc = np.mean(np.logical_and(box_matches, keypoint_matches))  # 修改：将 & 替换为 np.logical_and
    with open("...", "w") as file_1:
        # 确保将 id_list 中的所有元素转换为字符串
        file_1.write(" ".join(map(str, iou_all_scores)))  # 用空格连接每个元素素
    # 计算 IoU 平均值
    iou_all_mean = np.mean(iou_all_scores)
    iou_matched_mean = np.mean(iou_matched_scores)

    return box_accuracy, keypoint_accuracy, both_acc, iou_all_mean, iou_matched_mean, iou_all_scores, iou_matched_scores,  boxes_c_i


if __name__ == "__main__":
    # 准备真实框和关键点（此处需要根据实际情况填充）
    true_boxes = four_xy  # 从文件读取的真实框
    true_keypoints = ten_xy  # 从文件读取的真实关键点

    all_pred_boxes = []
    all_pred_keypoints = []
    m = []
    n = 0
    for image_path in args.image_paths:
        n += 1
        print(f"处理图像: {image_path}")
        boxes_c, landmarks = infer_image(image_path)
        if boxes_c is not None and len(boxes_c) > 0 and np.array_equal(boxes_c[0][0:4], [0, 0, 0, 0]):
            m.append(n)
            print("检测框为空")
        else:
            print("检测框有效")

        if landmarks is not None and len(landmarks) > 0 and np.array_equal(landmarks[0:10], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            print("未检测到人脸.")
            continue

        all_pred_boxes.append(boxes_c)
        all_pred_keypoints.append(landmarks)
        # print(f"框框box:{all_pred_boxes}")
        # print(f"关键点box:{all_pred_keypoints}")

    # 评估模型表现
    overall_box_accuracy, overall_keypoint_accuracy, both_accuracy_above_80, iou_all_mean, iou_part_mean, iou_all, iou_part,  boxes_c_i = \
        (evaluate_batch(true_boxes, all_pred_boxes, true_keypoints, all_pred_keypoints))
    print(f"框的整体准确率: {overall_box_accuracy}")
    print(f"关键点的整体准确率: {overall_keypoint_accuracy}")
    print(f"框和关键点都在线的比例: {both_accuracy_above_80}")
    print(f"总平均iou：{iou_all_mean}")
    print(f"大于0.5的测试图像的平均iou：{iou_part_mean}")
    print(f"平均置信度：{np.mean(boxes_c_i)}")
    with open("...", "w") as file_1:
        # 确保将 id_list 中的所有元素转换为字符串
        file_1.write(" ".join(map(str, m)))  # 用空格连接每个元素素
