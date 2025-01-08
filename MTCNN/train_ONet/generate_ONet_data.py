import os
import pickle
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_format_converter import convert_data
from utils.utils import py_nms, combine_data_list, crop_landmark_image, delete_old_img, pad, processed_image
from utils.utils import save_hard_example, generate_bbox, read_annotation, convert_to_square, calibrate_box
from utils.utils import get_landmark_from_lfw_neg, get_landmark_from_celeba

# 模型路径
model_path = '...'

device = torch.device("cuda")
# 获取P模型
pnet = torch.load(os.path.join(model_path, 'PNet.pth'))
pnet.to(device).half()  # 转为半精度浮点数
pnet.eval()
softmax_p = torch.nn.Softmax(dim=0)

# 获取R模型
rnet = torch.load(os.path.join(model_path, 'RNet.pth'))
rnet.to(device).half()  # 转为半精度浮点数
rnet.eval()
softmax_r = torch.nn.Softmax(dim=-1)



# 对比度调整函数（矢量化实现）
def adjust_contrast(image, low, high):
    if len(image.shape) != 2:
        raise ValueError("输入图像必须是灰度图像。")
    adjusted_image = np.clip(255 * (image - low) / (high - low), 0, 255).astype(np.uint8)
    adjusted_image[image < low] = 0
    adjusted_image[image > high] = 255
    return adjusted_image

# 新增: 图像预处理函数
def preprocess_image(image):  # 新增
    if len(image.shape) == 3:  # 如果是彩色图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 新增
    return image  # 新增

# 新增: 结合 ELBP 和 DLBP 特征提取函数
def extract_combined_elbp_dlbp(image, window_size=(3, 3), low=120, high=200):  # 新增
    image = preprocess_image(image)  # 新增
    image = (image * 255).astype(np.uint8)  # 新增
    adjusted_image = adjust_contrast(image, low, high)  # 新增
    combined_image = np.zeros_like(adjusted_image, dtype=np.uint16)  # 新增

    height, width = adjusted_image.shape  # 新增
    for y in range(0, height - window_size[0] + 1, window_size[0]):  # 新增
        for x in range(0, width - window_size[1] + 1, window_size[1]):  # 新增
            window = adjusted_image[y:y + window_size[0], x:x + window_size[1]]  # 新增

            center = window[1, 1]  # 新增
            i_max = np.max(window)  # 新增
            i_aver = np.mean(window)  # 新增
            i_T = i_max - i_aver  # 新增

            elbp_value = 0  # 新增
            dlbp_value = 0  # 新增
            for p in range(8):  # 新增
                neighbor = window[p // 3, p % 3]  # 新增
                if (int(neighbor) - int(center)) >= 0:  # 新增
                    elbp_value |= (1 << p)  # 新增
                if (int(neighbor) - int(center)) >= i_T:  # 新增
                    dlbp_value |= (1 << p)  # 新增

            combined_image[y + 1, x + 1] = elbp_value | (dlbp_value << 8)  # 新增

    return combined_image  # 修改为仅返回合成图像


# 使用PNet模型预测
def predict_pnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float16).half()  # 使用半精度
    infer_data = torch.unsqueeze(infer_data, dim=0)
    infer_data = infer_data.to(device)
    # 执行预测
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用RNet模型预测
def predict_rnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float16).half()  # 使用半精度
    infer_data = infer_data.to(device)
    # 执行预测
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


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
    cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
    if int(num_boxes) == 0:
        print('P模型检测结果为空！')
        return None, None
    for i in range(int(num_boxes)):
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24))
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
        return None, None

    keep = py_nms(boxes, 0.6, mode='Union')
    boxes = boxes[keep]
    # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes, boxes_c


# 截取pos,neg,part三种类型图片并resize成24x24大小作为RNet的输入
def crop_48_box_image(data_path, filename, min_face_size, scale_factor, p_thresh, r_thresh):
    # pos，part,neg裁剪图片放置位置
    pos_save_dir = os.path.join(data_path, '48/positive')
    part_save_dir = os.path.join(data_path, '48/part')
    neg_save_dir = os.path.join(data_path, '48/negative')
    # RNet数据地址
    save_dir = os.path.join(data_path, '48/')

    # 创建文件夹
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    # 读取标注数据
    data = read_annotation(data_path, filename)
    all_boxes = []
    landmarks = []
    empty_array = np.array([])

    # 使用PNet模型识别图片
    for image_path in tqdm(data['images']):
        assert os.path.exists(image_path), 'image not exists'
        im = cv2.imread(image_path)
        im = extract_combined_elbp_dlbp(im)
        while len(im.shape) == 3:  # 彩色图像
            im = extract_combined_elbp_dlbp(im)
        if len(im.shape) == 2:  # 灰度图像
            # 将灰度图像扩展到3通道
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        else:
            raise ValueError("不支持的图像格式。")
        boxes_c = detect_pnet(im, min_face_size, scale_factor, p_thresh)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue

        boxes, boxes_c = detect_rnet(im, boxes_c, r_thresh)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue

        all_boxes.append(boxes_c)

    # 把识别结果存放在文件中
    save_file = os.path.join(save_dir, 'detections.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, 1)

    save_hard_example(data_path, 48)


if __name__ == '__main__':
    data_path = '...'
    base_dir = '...'
    filename = '...'
    min_face_size = 20
    scale_factor = 0.79
    p_thresh = 0.6
    r_thresh = 0.7
    # 获取人脸的box图片数据
    print('开始生成bbox图像数据')
    crop_48_box_image(data_path, filename, min_face_size, scale_factor, p_thresh, r_thresh)
    # 获取人脸关键点的数据
    print('开始生成landmark图像数据')
    # 获取lfw negbox，关键点
    lfw_neg_path = os.path.join(data_path, 'trainImageList.txt')
    data_list = get_landmark_from_lfw_neg(lfw_neg_path, data_path)
    # 获取celeba，关键点
    # celeba_data_list = get_landmark_from_celeba(data_path)
    # data_list.extend(celeba_data_list)
    crop_landmark_image(data_path, data_list, 48, argument=True,  M="48")
    # 合并数据列表
    print('开始合成数据列表')
    combine_data_list(os.path.join(data_path, '48'))
    # 合并图像数据
    print('开始合成图像文件')
    convert_data(os.path.join(data_path, '48'), os.path.join(data_path, '48', 'all_data'))
    # 删除旧数据
    print('开始删除就得图像文件')
    delete_old_img(data_path, 48,  M="48")
