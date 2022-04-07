import numpy
import os
import json
import matplotlib.pyplot as plt
def lagrange(x, y, num_points, x_test):
    # 所有的基函数值，每个元素代表一个基函数的值
    l = numpy.zeros(shape=(num_points, ))

    # 计算第k个基函数的值
    for k in range(num_points):
        # 乘法时必须先有一个值,由于l[k]肯定会被至少乘n次，所以可以取1
        l[k] = 1
        # 计算第k个基函数中第k_个项（每一项：分子除以分母）
        for k_ in range(num_points):
            if k != k_:
                # 基函数需要通过连乘得到
                l[k] = l[k]*(x_test-x[k_])/(x[k]-x[k_])
            else:
                pass
    # 计算当前需要预测的x_test对应的y_test值
    L = 0
    for i in range(num_points):
        # 求所有基函数值的和
        L += y[i]*l[i]
    return L


# 读取文件名称和内容
def deal_files(joint_num,read_json):

    files = os.listdir(read_json) # 获取read_path下的所有文件名称（顺序读取的）

    x_t = numpy.zeros((18, 63), dtype=numpy.float)
    y_t = numpy.zeros((18, 63), dtype=numpy.float)

    t = 0
    t_aix = []
    n = [joint_num]
    a = 0

    for file_name in files:
        if a/2 == 1:
            a = a+1
            continue
        with open(read_json +"\\"+file_name,'r') as load_f:
            load_dict = json.load(load_f)
            for i in range(len(load_dict.get("people"))):
                x = []
                y = []
                c = []
                arr = load_dict.get("people")[i].get("pose_keypoints_2d")
                j = 0
                while j < len(arr):
                    x.append(arr[j])
                    y.append(arr[j + 1])
                    c.append(arr[j + 2])
                    j = j + 3

                for k in n:
                    # print(i,k)
                    if x[k] ==0:
                        x_t[k][t] = None
                    else:
                        x_t[k][t] = x[k]
                    if  y[k] == 0:
                        y_t[k][t] = None
                    else:
                        y_t[k][t] = y[k]
            a = a+1
        t = t + 1

    for j in range(t):
        t_aix.append(j)

    x = x_t[joint_num]
    y = y_t[joint_num]

    for fr in range(t):
        if x[fr] ==0:
            if x[fr+1] ==0:
                x_test = list(numpy.linspace(x[0], x[-1], 50))
                y_predict = [lagrange(x, y, len(x), x_i) for x_i in x_test]
            else:
                x[fr] = (x[fr - 1] + x[fr + 1]) / 2

    return x, y_predict