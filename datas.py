#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 开发人员：123
# 开发时间：2021/3/31  20:12
# 文件名称：datas
# 开发工具：PyCharm
import numpy as np
import os
# 主要进行文件类属性的获取
import cv2
import xml.etree.ElementTree as ET
import copy
# copy函数类似于使得地址空间相同并没有实际进行开拓数据空间，想要得到一份独立的数据
# 使用deepcopy函数


# 加载数据
class load_data(object):
    def __init__(self,path,batch,CLASS):
        self.devkil_path = path + '/VOCdevkit'
        self.data_path = self.devkil_path+'/VOC2007'
        self.img_size = 448
        self.batch = batch
        # 传入的数据 一次网络训练读入的数据大小
        self.CLASS = CLASS
        self.id  = 0

        # 字典格式，均可以成为其对象
        self.n_class = len(CLASS)
        self.class_id = dict(zip(CLASS, range(self.n_class)))
        # zip 实现将class类别和对应的范围打包成元组列表,range(n) 返回一个可迭代对象只是可以用作列表使用

        self.run_this()

    # 加载图像
    def load_img(self, PATH ):
        im = cv2.imread(PATH)
        # 对图像进行预处理
        im = cv2.resize(im, (self.img_size, self.img_size))
         # cv2 进行图像加载过程中是bgr 需要进行转换为RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # 进行归一化操作
        im = np.multiply(1./255., im)
        return im

# 加载xml文件
    def load_xml(self, index):
        path = self.data_path+'/JPEGImages/'+index+'.jpg'
        xml_path = self.data_path+'/Annotations/'+index+'.xml'
        img = cv2.imread(path)
        w = self.img_size / img.shape[0]
        h = self.img_size / img.shape[1]
        label = np.zeros((7,7,25))
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        for i in objs:
            box = i.find('bndbox')
            # 避免取出数据在图片范围内
            x1 = max(min((float(box.find('xmin').text)-1)*w, self.img_size-1), 0)
            # .text 去除标签，获取标签中的内容
            y1 = max(min((float(box.find('xmax').text)-1)*h, self.img_size-1), 0)
            # 注意括号位置，避免出现str-int 情况出现
            x2 = max(min((float(box.find('ymin').text)-1)*w, self.img_size-1), 0)
            y2 = max(min((float(box.find('ymax').text)-1)*h, self.img_size-1), 0)
            boxes = [(x1+x2)/2., (y1+y2)/2., (x2-x1), (y2-y1)]
            cls_id = self.class_id[i.find('name').text.lower().strip()]
            # lower函数将大写字母改为小写  strip函数去除字符串两边的空白字符比如\n\t\r
            x_id = int(boxes[0]*7/self.img_size)
            y_id = int(boxes[1]*7/self.img_size)
            if label[y_id, x_id, 0] == 1:
                continue
            label[y_id, x_id, 0] = 1  # 落在那个网格，置信度
            label[y_id, x_id, 1:5] = boxes
            label[y_id, x_id, 5 + cls_id] = 1 # 对应的类别编码为1
        return label, len(objs)   #len(objs)决定了目标标签的个数以及标签的实际维度

    def load_label(self):
        path = self.data_path+'/ImageSets/Main/trainval.txt'
        with open(path,'r') as f:
            index = [x.strip() for x in f.readlines()]
        labels = []
        for i in index:
            la, num = self.load_xml(i)
            if num == 0:
                continue
            img_name = self.data_path + '/JPEGImages/' + i + '.jpg'
            labels.append({'img_name': img_name,
                           'label': la})
        return labels
    def run_this(self):
        labels = self.load_label()
        np.random.shuffle(labels)
        # 将标签随机打乱
        self.truth_labels = labels
        return labels
    # 获取数据
    def get_data(self):
        img = np.zeros((self.batch,self.img_size,self.img_size,3))
        labels = np.zeros((self.batch,7,7,25))
        times = 0
        while times < self.batch:
            img_name = self.truth_labels[self.id]['img_name']
            # 对于相同的名字 按照存放的顺序依次取值。 truth_labels 并没有定义为任何类型,只会拿到名字
            img[times,:,:,:] = self.load_img(img_name)
            # 对应的times 开始那一块全部赋值进来
            labels[times, :, :, :] = self.truth_labels[self.id]['label']
            times += 1
            self.id += 1
            # 已经取完标签 重新打乱 初始化为零，可要可不要
            if self.id > len(self.truth_labels):
                np.random.shuffle(self.truth_labels)
                times = 0
                self.id = 0
        return img, labels

if __name__ == '__main__':
    # 相等才运行  ，调用时并不运行
    """
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

    data_path = '../data'
    # ./ 当前目录 ../ 父目录 /根目录
    test = load_data(data_path, 10, CLASSES)
    img, labels = test.get_data()
    print(img.shape, labels.shape)
    #  'NoneType' object has no attribute 'shape'   cv2读图片路径不对，注意大小写
    """
    img = cv2.imread("../data/VOCdevkit/VOC2007/JPEGImages/000005.jpg")
    cv2.imshow("im", img)
    img = img[:,::-1,:]
    # 实现水平翻转  宽度 长度  颜色通道   任意一个位置进行上述操作  形成不同效果 1 上下翻转，左右翻转   颜色翻转
    cv2.imshow("xx", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
