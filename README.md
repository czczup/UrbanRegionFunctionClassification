# 城市区域功能分类
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
把压缩文件放在data文件夹里直接解压应该就是上面这样

### 1.3 数据转换
把visit数据转换为矩阵，这一步耗时比较长，大概要一个小时
```
python visit2array.py
```

### 1.4 生成tfrecord
```
python tfrecord.py
```

### 1.5 训练
```
python train.py
```

### 1.6 测试
```
python test.py
```