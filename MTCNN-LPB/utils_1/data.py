import mmap

import cv2
import numpy as np
from torch.utils.data import Dataset


class ImageData(object):
    def __init__(self, data_path):
        self.offset_dict = {}
        for line in open(data_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(data_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('正在加载数据标签...')
        # 获取label
        self.label = {}
        self.box = {}
        self.landmark = {}
        label_path = data_path + '.label'
        for line in open(label_path, 'rb'):
            key, bbox, landmark, label = line.split(b'\t')
            self.label[key] = int(label)
            self.box[key] = [float(x) for x in bbox.split()]
            self.landmark[key] = [float(x) for x in landmark.split()]
        print('数据加载完成，总数据量为：%d' % len(self.label))

    # 获取图像数据
    def get_img(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    # 获取图像标签
    def get_label(self, key):
        return self.label.get(key)

    # 获取人脸box
    def get_bbox(self, key):
        return self.box.get(key)

    # 获取关键点
    def get_landmark(self, key):
        return self.landmark.get(key)

    # 获取所有keys
    def get_keys(self):
        return self.label.keys()

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


def process(image):
    image = np.fromstring(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_COLOR
    # 如果是3通道则转换为4通道
    if image is not None and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)  # 转换为RGBA
        image[..., 3] = extract_combined_elbp_dlbp(image)


    assert (image is not None), 'image is None'

    # 把图片转换成numpy值
    image = np.array(image).astype(np.float32)
    # 转换成CHW
    image = image.transpose((2, 0, 1))
    # 归一化
    image = (image - 127.5) / 128
    return image


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_path):
        super(CustomDataset, self).__init__()
        self.imageData = ImageData(data_path)
        self.keys = self.imageData.get_keys()
        self.keys = list(self.keys)
        np.random.shuffle(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = self.imageData.get_img(key)
        assert (img is not None)
        label = self.imageData.get_label(key)
        assert (label is not None)
        bbox = self.imageData.get_bbox(key)
        landmark = self.imageData.get_landmark(key)
        img = process(img)
        label = np.array([label], np.int64)
        bbox = np.array(bbox, np.float32)
        landmark = np.array(landmark, np.float32)
        return img, label, bbox, landmark

    def __len__(self):
        return len(self.keys)