# 城市区域功能分类

## 简介

![模型结构图](https://s2.ax1x.com/2019/05/05/E0PqnP.png)

同时使用遥感图像和访问数据两个模态，在特征层进行融合，大概能拿到0.55的准确率。

PS：刚刚测试了一下，有0.57分。

![tensorboard](https://s2.ax1x.com/2019/05/05/E0Puef.png)

## 快速起步
### 1.1 依赖的库
```
tensorflow-gpu==1.8
opencv-python
pandas 
```
### 1.2 数据准备
将数据放在data文件夹下，如下所示：
- data/test_image/test/xxxxxx.jpg
- data/test_visit/test/xxxxxx.txt
- data/train_image/train/00x/xxxxxx_00x.jpg
- data/train_visit/xxxxxx_00x.txt

把压缩文件放在data文件夹里直接解压应该就是上面这样。

我把给的训练集划分了一部分当验证集，具体过程看check_data.ipynb。

划分后的文件名记录在data/train.txt和data/valid.txt中。

### 1.3 数据转换
把visit数据转换为7x26x24的矩阵，这一步耗时比较长，大概要一个小时。
```
python visit2array.py
```
转换后的数据存储在:
- data/npy/train_visit
- data/npy/test_visit

### 1.4 生成tfrecord
```
python tfrecord.py
```
生成的tfrecord存储在：
- data/tfrecord/train.tfrecord
- data/tfrecord/valid.tfrecord

备注：由于这里直接加载了所有数据，大约要占用5G内存。

### 1.5 训练
```
python train.py
```
为了调参方便，每组实验存在不同的文件夹里。
需要输入显卡的编号和文件夹名称，比如：
```
device id: 0
dir id: 1001
```

查看tensorboard：
```
cd model/
tensorboard --logdir=./
```

### 1.6 测试
```
python test.py
```
测试完成后在result文件夹中生成结果。

![实测分数](https://s2.ax1x.com/2019/05/05/E0PYyq.png)

