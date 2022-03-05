#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import mxnet as mx
#from mxnet import memonger
import logging
import datetime
import numpy as np
import os
import sys
import CustomImage as CImage 
import random
import math

# Joy used setting
os.environ['MXNET_USE_FUSION'] = '0'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

def res_block(data,num_r, layer):
    num_r1 = int(num_r * (2. / 3.))
    slice_r = mx.symbol.SliceChannel(data = data, num_outputs = 3, name = ('slice%s_res' % layer))
    efm_r_max1 = mx.symbol.maximum(slice_r[0], slice_r[1])
    efm_r_min1 = mx.symbol.minimum(slice_r[0], slice_r[1])
    efm_r_max2 = mx.symbol.maximum(slice_r[2], efm_r_max1)
    efm_r_min2 = mx.symbol.minimum(slice_r[2], efm_r_min1)
    efm_r = mx.symbol.Concat(efm_r_max2, efm_r_min2)
    bn6 = mx.symbol.BatchNorm(data=efm_r, name=('bnr1-%s' % (layer)))
    conv_r = mx.symbol.Convolution(data=bn6, num_filter=num_r , dilate=(1, 1), kernel=(3, 3), name=('conv%s_res1' % layer), pad=(1, 1))
    slice_r = mx.symbol.SliceChannel(data=conv_r, num_outputs=3, name=('slice%s_res' % layer))
    efm_r_max1 = mx.symbol.maximum(slice_r[0], slice_r[1])
    efm_r_min1 = mx.symbol.minimum(slice_r[0], slice_r[1])
    efm_r_max2 = mx.symbol.maximum(slice_r[2], efm_r_max1)
    efm_r_min2 = mx.symbol.minimum(slice_r[2], efm_r_min1)
    efm_r = mx.symbol.Concat(efm_r_max2, efm_r_min2)
    bn61 = mx.symbol.BatchNorm(data=efm_r, name=('bnr2-%s' % (layer)))
    conv_r1 = mx.symbol.Convolution(data=bn61, num_filter=num_r1, dilate=(1, 1), kernel=(3, 3), name=('conv%s_res_r2' % layer), pad=(1, 1))
    conv_r0 = data + conv_r1
    return conv_r0

def group(data, num_r, num, kernel, stride, pad, layer, tar_num=0):
    if num_r > 0:
        if num_r % 3 == 0:
            if tar_num >= 1:
                res = res_block(data, num_r,layer)
            if tar_num >= 2:
                for x in range(1, tar_num):
                    res=res_block(res, num_r, layer + str(x))

            bn4 = mx.symbol.BatchNorm(data=res, name=('bn3-%s' % (layer)))
            conv_r2 = mx.symbol.Convolution(data=bn4, num_filter=num_r, dilate=(1, 1), kernel=(1, 1), name=('conv%s_r' % layer))
            slice_r = mx.symbol.SliceChannel(data=conv_r2, num_outputs=3, name=('slice%s_r' % layer))
            efm_r_max1 = mx.symbol.maximum(slice_r[0], slice_r[1])
            efm_r_min1 = mx.symbol.minimum(slice_r[0], slice_r[1])
            efm_r_max2 = mx.symbol.maximum(slice_r[2], efm_r_max1)
            efm_r_min2 = mx.symbol.minimum(slice_r[2], efm_r_min1)
            mfm_r = mx.symbol.Concat(efm_r_max2, efm_r_min2)
        else:
            bn5 = mx.symbol.BatchNorm(data=data, name=('bn4-%s' % (layer))) 
            conv_r = mx.symbol.Convolution(data=bn5, num_filter=num_r, dilate=(1, 1), kernel=(1, 1), name=('conv%s_r' % layer))
            slice_r = mx.symbol.SliceChannel(data=conv_r, num_outputs=2, name=('slice%s_r' % layer))
            mfm_r = mx.symbol.maximum(slice_r[0], slice_r[1])
        bn7 = mx.symbol.BatchNorm(data=mfm_r, name=('bn5-%s' % (layer)))
        conv = mx.symbol.Convolution(data=bn7, kernel=kernel, stride=stride, pad=pad, num_filter=num, dilate=(1, 1), name=('conv%s' % layer))
    else:   
        bn8 = mx.symbol.BatchNorm(data=data, name=('bn6-%s' % (layer)))
        conv = mx.symbol.Convolution(data=bn8, kernel=kernel, stride=stride, pad=pad, num_filter=num, dilate=(1, 1), name=('conv%s' % layer))
    if num % 3 == 0:
        slice = mx.symbol.SliceChannel(data=conv, num_outputs=3, name=('slice%s' % layer))
        mfm_max1 = mx.symbol.maximum(slice[0], slice[1])
        mfm_max2 = mx.symbol.maximum(mfm_max1, slice[2])
        mfm_min1 = mx.symbol.minimum(slice[0], slice[1])
        mfm_min2 = mx.symbol.minimum(mfm_min1, slice[2])
        mfm_maxpool = mx.symbol.Pooling(data=mfm_max2, pool_type="max", kernel=(2, 2), stride=(2, 2), name=('mfm_maxpool%s' % layer))
        mfm_min2 = -1 * mfm_min2
        mfm_minpool = mx.symbol.Pooling(data=mfm_min2, pool_type="max", kernel=(2, 2), stride=(2, 2), name=('mfm_minpool%s' % layer))
        mfm_invpool = -1 * mfm_minpool
        mfm = mx.symbol.Concat(mfm_maxpool, mfm_invpool)
    else:
        slice = mx.symbol.SliceChannel(data=conv, num_outputs=2, name=('slice%s' % layer))
        mfm = mx.symbol.maximum(slice[0], slice[1])
    return mfm

def Model_Build(data, classes):
    clabel = mx.symbol.Variable('softmax_label')
    #rlabel = mx.symbol.Variable('lin_reg_label')
    #data = mx.symbol.Variable(name="data")
    #print(data)
    pool1 = group(data, 0, 99, (5,5), (1,1), (2,2), str(1))
    pool2 = group(pool1, 99, 198, (3,3), (1,1), (1,1), str(2), 1)
    pool3 = group(pool2, 198, 387, (3,3), (1,1), (1,1), str(3), 2)
    pool4 = group(pool3, 387, 261, (3,3), (1,1), (1,1), str(4), 3)
    pool5 = group(pool4, 261, 261, (3,3), (1,1), (1,1), str(5), 4)
    bn_fd = mx.symbol.BatchNorm(data=pool5, name="bn_fd")
    global_pooling = mx.symbol.Pooling(data=bn_fd, pool_type="avg", global_pool=True, name="glopol")    

    flatten1 = mx.symbol.Flatten(data=global_pooling)
    fc1 = mx.symbol.FullyConnected(data=flatten1, num_hidden=513, name="fc1")
    
    slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=3, name="slice_fc1")
    efm_fc_max1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
    efm_fc_min1 = mx.symbol.minimum(slice_fc1[0], slice_fc1[1])
    efm_fc_max2 = mx.symbol.maximum(slice_fc1[2], efm_fc_max1)
    efm_fc_min2 = mx.symbol.abs(mx.symbol.minimum(slice_fc1[2], efm_fc_min1))
    efm_fc1 = mx.symbol.Concat(efm_fc_max2, efm_fc_min2)
    drop = mx.symbol.Dropout(data=efm_fc1, p=0.4, name="drop")
    flatten2 = mx.symbol.Flatten(data=drop) 
    #fc2 = mx.symbol.FullyConnected(data = flatten, num_hidden = classes, name = "fc2")
    #softmax = mx.symbol.SoftmaxOutput(data = fc2, label=clabel, name = 'softmax') # cross-entropy loss

    fc3 = mx.symbol.FullyConnected(data=flatten2, num_hidden=1, name="fc3")
    #freqact = mx.symbol.Activation(data=fc3, act_type='relu', name = "freqact")
    #freq = mx.symbol.LinearRegressionOutput(data=freqact, label=rlabel, name = "freq") # square loss
    #out = mx.symbol.Group([softmax, freq])
    freqact = mx.symbol.Activation(data=fc3, act_type='relu', name="freqact")
    out = mx.symbol.SoftmaxOutput(data=freqact, label=clabel, name="out") # softmax loss
    
    output = mx.symbol.Group([out, fc1])
    return output

def get_net(classes, margin=0.2):
    anc = mx.sym.Variable('anc_data')
    pos = mx.sym.Variable('pos_data')
    neg = mx.sym.Variable('neg_data')
    label = mx.sym.Variable('label')
    label = mx.sym.Reshape(data=label, shape=(-1, 1))

    anc_out = Model_Build(anc, classes)
    pos_out = Model_Build(pos, classes)
    neg_out = Model_Build(neg, classes)
    '''
    digraph = mx.viz.plot_network(output, save_format='png')
    digraph.view()
    '''
    fs = anc_out[1] - pos_out[1]
    fd = anc_out[1] - neg_out[1]
    fs = fs * fs
    fd = fd * fd
    fs = mx.sym.sum(fs, axis=1, keepdims=1)
    fd = mx.sym.sum(fd, axis=1, keepdims=1)
    loss = fd - fs
    loss = label - loss
    loss = mx.sym.Activation(data=loss, act_type='relu')
    triplet_loss = mx.sym.MakeLoss(loss)
    return triplet_loss

'''
def RMSE(label, pred):
    ret = 0.0
    n = 0.000000000001
    global batch_size
    #for k in range(pred.shape[1]):   #original line
    # Joy revision below
    for k in range(batch_size):
        v1 = label[k]
        v2 = pred[k][0]
        ret += (v1 - v2) * (v1-v2)
        n += 1.0
    return math.sqrt(ret / n)
'''

def Count_Img_num(Lst_Path, name):
    IMG_number = 0
    file = open(Lst_Path, "r")
    for linen in file:
        IMG_number += 1
    file.close()
    print("Total number of {} samples = {}".format(name, IMG_number))
    return IMG_number

def cosine_dist(a, b):
    a1 = mx.nd.expand_dims(a, axis=1)   # (1,28,28) -> (1,1,28,28)
    b1 = mx.nd.expand_dims(b, axis=0)   # (1,28,28) -> (1,1,28,28)
    #d = mx.nd.batch_dot(a1, b1).squeeze()   # (1,1,28,28) -> (28, 28)
    #a_norm = mx.nd.sqrt(mx.nd.sum(mx.nd.square(a1)))    # scalar
    #b_norm = mx.nd.sqrt(mx.nd.sum(mx.nd.square(b1)))    # scalar
    #cos = d / (a_norm * b_norm) # (28,28)
    a2 = mx.nd.flatten(a).asnumpy()
    b2 = mx.nd.flatten(b).asnumpy()
    cos = mx.nd.array(1 - spatial.distance.cosine(a2, b2))
    #dist = mx.nd.sqrt(mx.nd.sum(mx.nd.square(b1 - a1)))  #scalar
    return cos

class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('Auc')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)

# create a dataset of positive image for each identities
def define_pos(data_iter, length, batch_size):
    pos_img = {}
    for epoch in range(length):
        for batch in data_iter:
            for i in range(batch_size):
                if int(batch.label[0][i].asscalar()) not in pos_img:
                    pos_img[int(batch.label[0][i].asscalar())] = copy.copy(batch.data[0][i])
    # pos_img: {1: img1, 2: img2, ...}
    return pos_img

class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DataIter(mx.io.DataIter):
    def __init__(self, data_iter, length, pos_img, batch_size, dshape):
        super(DataIter, self).__init__()
        
        self.batch_size = batch_size
        self.length = length
        self.data_iter = data_iter
        self.pos_img = pos_img
        self.dshape = dshape
        self.provide_data = [('anc_data', self.dshape), \
                            ('pos_data', self.dshape), \
                            ('neg_data', self.dshape)]
        self.provide_label = [('label', (self.batch_size, 1))]
        
        self.batch, self.label = self.make_pairs(self.data_iter, self.pos_img, self.batch_size)

    def make_pairs(self, data_iter, pos_img, batch_size):
        dataset = []
        labels = []
        for batch in data_iter:
            for i in range(batch_size / 2):
                # dataset: [[anc1, pos1], [anc2, pos2], ...]
                dataset += [[batch.data[0][i].asnumpy(), pos_img[int(batch.label[0][i].asscalar())].asnumpy()]]
                labels += [batch.label[0][i].asnumpy()]
        print(len(dataset), len(labels))
        print(len(labels[0][0][0]))
        return dataset, labels
        
    def __iter__(self):
        print('begin...')
        start = 0
        for i in range(self.length):
            anc_data = []
            pos_data = []
            neg_data = []
            label = []
            for j in range(start, start + self.batch_size / 2):
                anc_data.append(self.batch[0][j])
                pos_data.append(self.batch[1][j])
                label.append(self.label[j])
                label.append(self.label[j])
            
            for k in range(start, start + self.batch_size / 2):
                j = start + random.randint(0, batch_size/2 - 1)
                while int(label[j].asscalar()) == int(label[k].asscalar()):
                    j = start + random.randint(0, batch_size - 1)
                neg_data.append(self.batch[0][j])

            data_all = [mx.nd.array(anc_data), mx.nd.array(pos_data), mx.nd.array(neg_data)]
            label_all = [mx.nd.array(label)]
            data_names = ['anc_data', 'pos_data', 'neg_data']
            label_names = ['label']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            print('load success!!!')
            start += (self.batch_size / 2)

            yield data_batch


""" load data from the path """
root = sys.argv[1]
Training_Lst_path = os.path.join(root, 'train.lst')
Testing_Lst_path = os.path.join(root, 'test.lst')
Training_Rec_path = os.path.join(root, 'train.rec') 
Testing_Rec_path  = os.path.join(root, 'test.rec')
Training_Idx_path = os.path.join(root, 'train.idx')
Testing_Idx_path  = os.path.join(root, 'test.idx')


Training_IMG_number = Count_Img_num(Training_Lst_path, 'training')
Testing_IMG_number = Count_Img_num(Testing_Lst_path, 'testing')

""" batch size, image size """
Training_IMG_Height = 128
Training_IMG_Width = 128
Training_IMG_channel = 1
Training_IMG_classes = 8398     # 79078 for celeb1m, 8398 for parts of asian_celeb
batch_size = 32

""" setting GPUs """
#devs = [mx.gpu(0),mx.gpu(1)]
devs = mx.gpu()

epoch_size = int(Training_IMG_number / batch_size)
dshape = (batch_size, Training_IMG_channel, Training_IMG_Height, Training_IMG_Width)
print("epoch_size = {}".format(epoch_size))


""" generate model & log file """
Log_save_dir = ensure_dir('./log/')
Model_save_dir = ensure_dir('./models/')
Model_save_name = 'EFM_Freq'


""" setting for log file, auto. record values, training time and accuracy """
logging.basicConfig(filename = Log_save_dir + Model_save_name + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".log", level = logging.INFO)
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)


""" setting data iterator """
train_dataiter = CImage.ImageIter(path_imgrec=Training_Rec_path, path_imgidx=Training_Idx_path ,batch_size=batch_size, data_shape=(Training_IMG_channel, Training_IMG_Height, Training_IMG_Width), shuffle=True, aug_list=CImage.CreateAugmenter((Training_IMG_channel, Training_IMG_Height,Training_IMG_Width), rand_crop=False, rand_mirror=False), label_width=1, label_name="softmax_label")
test_dataiter =  CImage.ImageIter(path_imgrec=Testing_Rec_path, path_imgidx=Testing_Idx_path ,batch_size=batch_size, data_shape=(Training_IMG_channel, Training_IMG_Height, Training_IMG_Width), aug_list=CImage.CreateAugmenter((Training_IMG_channel, Training_IMG_Height,Training_IMG_Width), rand_crop=False), label_width=1, label_name="softmax_label")


""" Triplet pairs """
print('defining positive image...')
# Store a positive image for each identities
pos_img_train = define_pos(train_dataiter, int(epoch_size), batch_size)
pos_img_test = define_pos(test_dataiter, int(Testing_IMG_number / batch_size), batch_size)

train_dataiter.reset()
test_dataiter.reset()

print('making training pairs...')
data_train = DataIter(train_dataiter, int(epoch_size), pos_img_train, batch_size, dshape)
print('making testing pairs...')
data_test = DataIter(test_dataiter, int(Testing_IMG_number / batch_size), pos_img_test, batch_size, dshape)


devs = [mx.gpu(0)]

print("Building Module...")
new_sym = get_net(Training_IMG_classes, margin=0.2)
#net_mem_planned = memonger.search_plan(new_sym, data=dshape)

mod = mx.mod.Module(symbol=new_sym[0], data_names=('data', ), label_names=('label', ), context=devs)
mod.bind(data_shapes=data_train.provide_data, label_shapes=data_train.provide_label)
mod.init_params(mx.init.Xavier(factor_type="in", magnitude=2.34))
print("Parameters have been initialized.")


###############################
#   load pre-trained model    #
###############################
#symb = mx.symbol.load("EFM_RES.json")
#mod = mx.mod.Module(symbol=symb, label_names=['softmax_label'], context=devs)
#print("Loading Module...")
#mod.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)
################################


lr = 0.00024
wd = 0.0001
num_epoch = 100


kv = mx.kvstore.create('local')
op = mx.optimizer.create('sgd', rescale_grad=(1.0 / batch_size), momentum=0.9, lr_scheduler=mx.lr_scheduler.FactorScheduler(step=int(epoch_size * 2), factor=0.917, stop_factor_lr=5e-15), learning_rate=lr, wd=wd)
checkpoint = mx.callback.do_checkpoint(Model_save_dir + Model_save_name)


print("Start trainig...")
metric = Auc()
# eval_metric = mx.metric.np(RMSE)
mod.fit(data_train, data_test, eval_metric=metric, num_epoch=num_epoch, batch_end_callback=mx.callback.Speedometer(batch_size, 100), kvstore=kv, optimizer=op, epoch_end_callback=checkpoint)


for epoch in range(epoch_size):
    data_train.reset()
    metric.reset()
    for batch in data_train:
        mod.forward(batch, is_train=True)
        mod.update_metric(metric, batch.label)
        mod.backward()
        mod.update()
    print("epoch %d: Training %s").format(epoch, metric.get())

print(mod.tojson())
mod.save(Model_save_dir + "/" + Model_save_name + ".json")         # To save model structure

