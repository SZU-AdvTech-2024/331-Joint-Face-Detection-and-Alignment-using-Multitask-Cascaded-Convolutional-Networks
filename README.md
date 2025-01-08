# 引言
本研究基于改进的LBP优化MTCNN，Multi-task convolutional neural network（多任务卷积神经网络）在复杂人脸遮挡场景的人脸检测，原MTCNN是通过三个级联的卷积神经网络（P-Net、R-Net、O-Net）来逐步精细化人脸检测，但面对复杂人脸遮挡场景时对面部局部区域的纹理和边缘信息提取不足，导致人脸检查具有一定局限性。因此本研究利用改进后的LBP对MTCNN进行改进，提升其在复杂人脸遮挡场景的人脸检测精度。

# 说明
本研究的代码实现是MTCNN（PyTorch版本）(https://github.com/yeyupiaoling/Pytorch-MTCNN)的基础上改进完成的。

# 环境
 - Pytorch 1.8.1
 - Python 3.7

# 文件介绍
 - `MTCNN-LBP/` 改进后的MTCNN的代码实现文件
   - `models_1/` 模型文件
     - `Loss.py` 所使用的损失函数
     - `PNet.py` 训练好的PNet
     - `RNet.py` 训练好的RNet
     - `ONet.py` 训练好的ONet
   - `train_PNet_1/` PNet的训练代码
     - `generate_PNet_data.py` 生成PNet训练的数据
     - `train_PNet.py` 训练PNet网络模型
   - `train_RNet_1/` RNet的训练代码
     - `generate_RNet_data.py` 生成RNet训练的数据
     - `train_ONet.py` 训练RNet网络模型
   - `train_PNet_1/` ONet的训练代码
     - `generate_ONet_data.py` 生成ONet训练的数据
     - `train_ONet.py` 训练ONet网络模型
   - `utils_1/` 工具函数文件
     - `data.py` 训练数据读取器
     - `data_format_converter.py` 数据集合处理
     - `utils.py` 各种工具函数

MTCNN文件夹中是原MTCNN的代码实现，与改进后的MTCNN代码实现类似

 - `infer_1.py` 改进后MTCNN的在 Deep Convolutional Network Cascade for Facial Point Detection 数据集中的图像预测数据统计
 - `infer_O.py` 原MTCNN的在 Deep Convolutional Network Cascade for Facial Point Detection 数据集中的图像预测数据统计
 - `infer_celebA_1.py` 改进后MTCNN的在 CelebA 数据集中的图像预测数据统计
 - `infer_celebA_O.py` 原MTCNN的在 CelebA 数据集中的图像预测数据统计
 - `infer_pic_1.py` 改进后MTCNN在单张图片上的预测与展示
 - `infer_pic_O.py` 原MTCNN在单张图片上的预测与展示



# 数据集下载
 - [WIDER Face：](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) 下载训练数据WIDER Face Training Images，解压的WIDER_train文件夹放置到dataset下。并下载 [Face annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip) ，解压把里面的 wider_face_train_bbx_gt.txt 文件放在dataset目录下，
 - [Deep Convolutional Network Cascade for Facial Point Detection：](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) 下载 Training set 并解压，将里面的 lfw_5590 和 net_7876 文件夹放置到dataset下
 - 解压数据集之后，`dataset`目录下应该有文件夹`lfw_5590`，`net_7876`，`WIDER_train`，有标注文件`testImageList.txt`，`trainImageList.txt`，`wider_face_train.txt`
 - [CelebA：](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 下载未对齐的数据集，并随机选取属性中戴眼镜和戴帽子的图片，复杂人脸遮挡场景的数据集


# 训练模型

训练模型一共分为三步，分别是训练PNet模型、训练RNet模型、训练ONet模型，每一步训练都依赖上一步的结果。
原MTCNN模型和改进后的MTCNN模型的训练都是一样的。

## 第一步 训练PNet模型
 - `generate_PNet_data.py` 首先需要生成PNet模型训练所需要的图像数据
 - `train_PNet.py` 开始训练PNet模型

## 第二步 训练RNet模型
 - `generate_RNet_data.py` 使用上一步训练好的PNet模型生成RNet训练所需的图像数据
 - `train_RNet.py` 开始训练RNet模型


## 第三步 训练ONet模型
 - `generate_ONet_data.py` 使用上两部步训练好的PNet模型和RNet模型生成ONet训练所需的图像数据
 - `train_ONet.py` 开始训练ONet模型

# 预测
 - `infer_celebA_1.py` 输入对应路径，得到改进后MTCNN的在 CelebA 数据集中的图像预测数据统计结果
 - `infer_celebA_O.py` 输入对应路径，得到原MTCNN的在 CelebA 数据集中的图像预测数据统计结果
 - `infer_pic_1.py` 输入对应路径，得到改进后MTCNN在单张图片上的预测与展示
 - `infer_pic_O.py` 输入对应路径，得到原MTCNN在单张图片上的预测与展示


## 参考资料

1. https://github.com/AITTSMD/MTCNN-Tensorflow
2. https://blog.csdn.net/qq_36782182/article/details/83624357
3. https://github.com/yeyupiaoling/Pytorch-MTCNN

## 主要参考文献

1. Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. *IEEE signal processing letters*, 23(10), 1499-1503. 
2. Zhang, N., Luo, J., & Gao, W. (2020, September). Research on face detection technology based on MTCNN. In *2020 international conference on computer network, electronic and automation (ICCNEA)* (pp. 154-158). IEEE. 
3. Wang, X., Cao, J., Hao, Q., Zhang, K., Wang, Z., & Rizvi, S. (2018). LBP-based edge detection method for depth images with low resolutions. *IEEE Photonics Journal*, 11(1), 1-11. 
4. Al-wajih, E., & Ghazali, R. (2021). An enhanced LBP-based technique with various size of sliding window approach for handwritten Arabic digit recognition. *Multimedia Tools and Applications*, 80, 24399-24418. 
5. Bao, Y., & Dang, R. (2021, November). Face detection under non-uniform low light based on improved mtcnn. In *2021 2nd International Conference on Artificial Intelligence and Computer Engineering (ICAICE)* (pp. 704-707). IEEE. 
