#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 开发人员：123
# 开发时间：2021/4/1  21:56
# 文件名称：yolo_test
# 开发工具：PyCharm
import numpy as np
import tensorflow as tf
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 后续的variable_scope 函数将不再支持  使用前使用tf.compat.v1.varba......
"""
    7 x 7 ,物体目标中心  --》 落入某个网格之中，那么这个网格负责检测这个物体
    每个网格 ---》 2 个box 和这些box的置信度  confidence【0,1】
    cls_score   score  = confidence x IOU  物体没有落在其中，score = 0
    全连接层
    fc = [batch,7,7,30]  30 : 2个box，（x,y,w,h,c）--->10
    20 --->类别的置信度
    



"""
class Yolo(object):

    def __init__(self):
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
        self.c = len(self.classes)
        self. x_offest = np.transpose(np.reshape(np.array([np.arange(7)]*7*2),
                                                        [2, 7, 7]), [1, 2, 0])
        # 块的形状也会发生变化
        # 注意这里中括号的位置，生成14个arange（7）数组 ， reshape将生成数组分成两块，都是7x7
        # transpose就是通过坐标索引变换，将值从一个坐标索引到另一个坐标中罢了。  默认坐标轴0,1,2
        self.y_offest = np.transpose(self.x_offest,[1,0,2])
        self.threshold = 0.2 # confidence scores
        self.iou_threshold = 0.5
        self.max_output_size = 10
        self.img_shape = (448, 448)

        # relu: f(x) = max(x, 0 )
    def leak_relu(self, x, alpha = 0.1):
        return tf.maximum(alpha*x, x)

    ### 网络

    def _conv_layer(self, x, num_filters, filter_size, stride, scope):
        in_channels = x.get_shape().as_list()[-1]
        # x.get_shape()，返回一个元组类型，as_list将结果转化为list类  x是tensorflow类型

        weight = tf.Variable(tf.truncated_normal([filter_size,filter_size,
                                                  in_channels, num_filters]
                                                  , stddev=0.1), name='weights')
        #  tf.truncated_normal函数返回指定形状的张量填充随机截断的正常值.
        #  tf.Variable 初始化使用
        bias = tf.Variable(tf.zeros([num_filters,]),name='biases')
        # num_filters 个零   name用于变量命名
        pad_size = filter_size // 2
        # // 向下取整
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        # 转换为矩阵
        x_pad = tf.pad(x, pad_mat)
        # 主要是对张量在各个维度上进行填充 tensor（first） 与 padding（second） 的rank 必须相同
        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID", name=scope)
        # strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，
        # 第一位和最后一位固定必须是1
        # valid 不考虑边界  same考虑边界，不足的地方使用0去填充


        output = self.leak_relu(tf.nn.bias_add(conv, bias))
        # 将bias 加到conv（value）上

        return output

    def _fc_layer(self, x, num_out, activation=None, scope=None):

        num_in = x.get_shape().as_list()[-1]
        # 取最后一个元素
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1), name='weights')
        bias = tf.Variable(tf.zeros([num_out, ]), name='biases')
        output = tf.nn.xw_plus_b(x, weight, bias, name=scope)
        #  相当于x*weight +bias   矩阵乘法
        if activation:
            # 激活函数
            output = activation(output)

        return output
    def _maxpool_layer(self, x, pool_size, stride):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                                # batch_size，height ，width，channels
                                strides = [1, stride, stride, 1], padding='SAME')
        return output
    def _flatten(self,x):
        tran_x = tf.transpose(x, [0, 3, 1, 2])
        nums = np.product(x.get_shape().as_list()[1:])
        # prod 数组内求乘积
        # 第二个到最后一个
        return tf.reshape(tran_x, [-1, nums])
        # 转换为 nmus 列





    # 1*1卷积核 宽高不会变化  channel的通道可以改变（图片的厚度）
    def _build_net(self): #  24个卷积    3 个全连接层
        x = tf.placeholder(tf.float32, [None, 448, 448, 3])
        #   datatype TensorFlow中加载图片的维度为[batch, height, width, channels]
        #  with 语句适用于对资源进行访问的场合，确保不管使用过程中
        # 是否发生异常都会执行必要的“清理”操作，释放资源，
        # 比如文件使用后自动关闭／线程中锁的自动获取和释放等
        with tf.variable_scope('yolo'):
            with tf.variable_scope('conv_2'):
                # 用于定义创建变量（层）的操作的上下文管理器
                net = self._conv_layer(x, 64, 7, 2, 'conv_2')
            net = self._maxpool_layer(net, 2, 2)

            with tf.variable_scope('conv_4'):
                net = self._conv_layer(net, 192, 3, 1, 'conv_4')
            net = self._maxpool_layer(net, 2, 2)
            with tf.variable_scope('conv_6'):
                net = self._conv_layer(net, 128, 1, 1, 'conv_6')
            with tf.variable_scope('conv_7'):
                net = self._conv_layer(net, 256, 3, 1, 'conv_7')
            with tf.variable_scope('conv_8'):
                net = self._conv_layer(net, 256, 1, 1, 'conv_8' )
            with tf.variable_scope('conv_9'):
                net = self._conv_layer(net, 512, 3, 1, 'conv_9')
            net = self._maxpool_layer(net, 2, 2)
            with tf.variable_scope('conv_11'):
                net = self._conv_layer(net, 256, 1, 1, 'conv_11')
            with tf.variable_scope('conv_12'):
                net = self._conv_layer(net, 512, 3, 1, 'conv_12')
            with tf.variable_scope('conv_13'):
                net = self._conv_layer(net, 256, 1, 1, 'conv_13')
            with tf.variable_scope('conv_14'):
                net = self._conv_layer(net, 512, 3, 1, 'conv_14')
            with tf.variable_scope('conv_15'):
                net = self._conv_layer(net, 256, 1, 1, 'conv_15')
            with tf.variable_scope('conv_16'):
                net = self._conv_layer(net, 512, 3, 1, 'conv_16')
            with tf.variable_scope('conv_17'):
                net = self._conv_layer(net, 256, 1, 1, 'conv_17')
            with tf.variable_scope('conv_18'):
                net = self._conv_layer(net, 512, 3, 1, 'conv_18')
            with tf.variable_scope('conv_19'):
                net = self._conv_layer(net, 512, 1, 1, 'conv_19')
            with tf.variable_scope('conv_20'):
                net = self._conv_layer(net, 1024, 3, 1, 'conv_20')
            net = self._maxpool_layer(net, 2, 2)
            with tf.variable_scope('conv_22'):
                net = self._conv_layer(net, 512, 1, 1, 'conv_22')
            with tf.variable_scope('conv_23'):
                net = self._conv_layer(net, 1024, 3, 1, 'conv_23')
            with tf.variable_scope('conv_24'):
                net = self._conv_layer(net, 512, 1, 1, 'conv_24')
            with tf.variable_scope('conv_25'):
                net = self._conv_layer(net, 1024, 3, 1, 'conv_25')
            with tf.variable_scope('conv_26'):
                net = self._conv_layer(net, 1024, 3, 1, 'conv_26')
            with tf.variable_scope('conv_28'):
                net = self._conv_layer(net, 1024, 3, 2, 'conv_28')
            with tf.variable_scope('conv_29'):
                net = self._conv_layer(net, 1024, 3, 1, 'conv_29')
            with tf.variable_scope('conv_30'):
                net = self._conv_layer(net, 1024, 3, 1, 'conv_30')
            net = self._flatten(net)
            with tf.variable_scope('fc_33'):
                net = self._fc_layer(net, 512, activation=self.leak_relu, scope='fc_33')
            with tf.variable_scope('fc_34'):
                net = self._fc_layer(net, 4096, activation=self.leak_relu, scope='fc_34')
            with tf.variable_scope('fc_36'):
                net = self._fc_layer(net, 7 * 7 * 30, scope='fc_36')
        return net, x
    #####   iou
    def filter(self,pred):
        # 拿到类别在reshape
        cls = tf.reshape(pred[0,: 7*7*20], [7,7,20])
            # 零行 从0取到7*7*20  数组索引没有写完  取前面已经默认的
        confidence = tf.reshape(pred[0, 7*7*20 : 7*7*20+7*7*2], [7,7,2])
        # 7*7grad 每个2个boundingbox
        boxes = tf.reshape(pred[0,7*7*20+7*7*2:],[7,7,2,4])
        # 关于数组 按照顺序分块 依次下去 两行四列
        # 真实的box值    数据有进行过  归一化  4.8
        boxes = tf.stack([
            (boxes[:,:,:,0]+ tf.constant(self.x_offest,dtype=tf.float32))/ 7*self.img_shape[0],
            (boxes[:,:,:,1]+ tf.constant(self.y_offest,dtype=tf.float32))/ 7*self.img_shape[1],
            tf.square(boxes[:, :, :, 2])*self.img_shape[0],
            tf.square(boxes[:, :, :, 3])*self.img_shape[1]
                ],axis = 3)
        # 使用stack维度会增加  变成1,7,7,2  ,一共四组将会变成 4,7,7,2, 加上axis= 3   变成7,7,2,4
        scores = tf.expand_dims(confidence,-1)*tf.expand_dims(cls,2)
        # 7，7，2，1 跟7，7，1，20  相乘按照正常逻辑相乘  变成7，7，2,20（属于点乘）
        # 扩展维度
        print(scores)
        scores = tf.reshape(scores, [-1, 20])
        boxes = tf.reshape(boxes, [-1, 4])
        # 每个box的得分
        box_cls = tf.argmax(scores,axis= 1)
        # 指定的维（或轴），取值为0，代表延列取最大值索引，取值为1，表示延行取最大值索引
        box_cls_scores = tf.reduce_max(scores,axis =  1)
        # 指定的维（或轴），取值为0，代表延列取最大值索引，取值为1，表示延行取最大值
        # guolv
        filter_mask = box_cls_scores >= self.threshold
        # 得到bool值的数组
        scores = tf.boolean_mask(box_cls_scores, filter_mask)
        # 将true对应的值拿出来返回一个数组
        boxes = tf.boolean_mask(boxes, filter_mask)
        # 一一对应 20个类别  坐标四个
        box_cls = tf.boolean_mask(box_cls, filter_mask)
        # 拿到索引

        # xmin，ymin，xmax，ymax
        box = tf.stack([
            boxes[:,0]-.5*boxes[:,2],
            boxes[:,1]-.5*boxes[:,3],
            boxes[:,0]+.5*boxes[:,2],
            boxes[:,1]+.5*boxes[:,3]
        ], axis = 1)
        # NMS
        nms = tf.image.non_max_suppression(box,scores,self.max_output_size,self.iou_threshold)
        # 获得可以选择的box索引
        # 选出最大分值 然后跟其他的依次做交并大于一定值 删去那个   依次做下去，知道没有元素可以选取
        scores = tf.gather(scores,nms)
        # 获得选取的索引值 （坐标位置）
        boxes = tf.gather(boxes, nms)
        box_cls = tf.gather(box_cls,nms)
        return scores, boxes,box_cls

    ###  处理img函数
    def handle_img(self,img):
        img = cv2.resize(img,(448,448))
        self.re_img = img
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img,0)
        img = np.multiply(1./255.,img)
        return img
    def draw_rectangle(self,img, classes, scores, bboxes, colors, thickness=2):

        for i in range(bboxes.shape[0]):
            x = int(bboxes[i][0])
            y = int(bboxes[i][1])
            w = int(bboxes[i][2]) // 2
            h = int(bboxes[i][3]) // 2

            print("[x, y, w, h]=[%d, %d, %d, %d]" % (x, y, w, h))


            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), colors[0], thickness)
            # Draw text...
            s = '%s/%.3f' % (self.classes[classes[i]], scores[i])
            cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, s, (x - w + 5, y - h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.namedWindow("img", 0);
        cv2.resizeWindow("img", 640, 480);
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def _detect_from_image(self, image):
        """Do detection given a cv image"""


        img_resized = cv2.resize(image, (448, 448))
        self.img = img_resized
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_RGB = np.expand_dims(img_RGB,0)


        img_resized_np = np.asarray(img_RGB)
        _images = np.zeros((1, 448, 448, 3), dtype=np.float32)
        _images[0] = (img_resized_np / 255.0) * 2.0 - 1.0


        return _images

if __name__ == '__main__':
    yolo = Yolo()
    pred, x = yolo._build_net()
    # print(net)
    # 7x7x30 = 1470
    # ("yolo/fc_36/fc_36:0", shape=(?, 1470), dtype=float32)

    #print(scores,boxes, box_cls)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    # 初始化全局变量
    saver = tf.train.Saver()
    ckpt_path = 'YOLO_small.ckpt'
    img = cv2.imread('../data/VOCdevkit/VOC2007/JPEGImages/000009.jpg')
    img = yolo.handle_img(img)

    scores, boxes, box_classes = yolo.filter(pred)
    sess.run(init)
    saver.restore(sess, ckpt_path)


    #img = yolo._detect_from_image(img)



    scores, boxes, box_classes = sess.run([scores, boxes, box_classes], feed_dict={x: img})
    yolo.draw_rectangle(yolo.re_img, box_classes, scores, boxes, [[0, 0, 255], [255, 0, 0]])