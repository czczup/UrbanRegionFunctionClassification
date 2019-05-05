from model import MultiModal
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2

# 选择gpu设备
deviceId = input("device id: ")
os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
# 选择文件夹
dirId = input("dir id: ")

# 加载训练好的模型
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with sess.graph.as_default():
    with sess.as_default():
        model = MultiModal()
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += [var for var in tf.global_variables() if "global_step" in var.name]
        var_list += tf.trainable_variables()
        saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
        last_file = tf.train.latest_checkpoint("../model/"+dirId)
        if last_file:
            tf.logging.info('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)

images = []
visits = []

# 载入所有测试数据
for i in range(10000):
    image = cv2.imread("../data/test_image/test/"+str(i).zfill(6)+".jpg", cv2.IMREAD_COLOR)[0:88,0:88,:] / 255.0
    visit = np.load("../data/npy/test_visit/"+str(i).zfill(6)+".npy")
    images.append(image)
    visits.append(visit)

predictions = []

# 每次测试1000条数据，如果显存不够可以改小一些
for i in range(10):
    predictions.extend(sess.run(tf.argmax(model.prediction, 1),
                          feed_dict={model.image: images[i*1000:i*1000+1000],
                                     model.visit: visits[i*1000:i*1000+1000],
                                     model.training: False}))
    print(i)

# 新建文件夹
if not os.path.exists("../result/"):
    os.mkdir("../result/")

# 将预测结果写入文件
f = open("../result/result.txt", "w+")
for index, prediction in enumerate(predictions):
    f.write("%s \t %03d\n"%(str(index).zfill(6), prediction+1))
f.close()
print("测试完成")
