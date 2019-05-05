import time
import numpy as np
import sys
import datetime
import pandas as pd
import os

# 用字典查询代替类型转换，可以减少一部分计算时间
date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i

# 访问记录内的时间从2018年10月1日起，共182天
# 将日期按日历排列
for i in range(182):
    date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
    date_int = int(date.__str__().replace("-", ""))
    date2position[date_int] = [i%7, i//7]
    datestr2dateint[str(date_int)] = date_int


def visit2array(table):
    strings = table[1]
    init = np.zeros((7, 26, 24))
    for string in strings:
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])
        for date, visit_lst in temp:
            # x - 第几周
            # y - 第几天
            # z - 几点钟
            # value - 到访的总人数
            x, y = date2position[datestr2dateint[date]]
            for visit in visit_lst: # 统计到访的总人数
                init[x][y][str2int[visit]] += 1
    return init


def visit2array_test():
    start_time = time.time()
    for i in range(0, 10000):
        filename = str(i).zfill(6)
        table = pd.read_table("../data/test_visit/test/"+filename+".txt", header=None)
        array = visit2array(table)
        np.save("../data/npy/test_visit/"+filename+".npy", array)
        sys.stdout.write('\r>> Processing visit data %d/%d'%(i+1, 10000))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("using time:%.2fs"%(time.time()-start_time))


def visit2array_train():
    table = pd.read_csv("../data/train.txt", header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    for index, filename in enumerate(filenames):
        table = pd.read_table("../data/train_visit/"+filename+".txt", header=None)
        array = visit2array(table)
        np.save("../data/npy/train_visit/"+filename+".npy", array)
        sys.stdout.write('\r>> Processing visit data %d/%d'%(index+1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("using time:%.2fs"%(time.time()-start_time))


def visit2array_valid():
    table = pd.read_csv("../data/valid.txt", header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    for index, filename in enumerate(filenames):
        table = pd.read_table("../data/train_visit/"+filename+".txt", header=None)
        array = visit2array(table)
        np.save("../data/npy/train_visit/"+filename+".npy", array)
        sys.stdout.write('\r>> Processing visit data %d/%d'%(index+1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("using time:%.2fs"%(time.time()-start_time))


if __name__ == '__main__':
    if not os.path.exists("../data/npy/test_visit/"):
        os.makedirs("../data/npy/test_visit/")
    if not os.path.exists("../data/npy/train_visit/"):
        os.makedirs("../data/npy/train_visit/")
    visit2array_train()
    visit2array_valid()
    visit2array_test()

