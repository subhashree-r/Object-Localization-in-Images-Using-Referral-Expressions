
from __future__ import absolute_import
from __future__ import print_function
import cv2
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

import numpy as np

import random
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils.vis_utils import plot_model
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.90

num_classes = 10


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x1_train,x2_train):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pair = []
    pairx1 =[]
    pairx2 =[]
    pairx1n=[]
    pairx2n=[]
    labels = []
    num_size = len(x1_train)
    for i in range(num_size):
        # pair+=[np.concatenate((np.array(x1_train[i]),np.array(x2_train[i])),axis=0)]
        pairx1.append(x1_train[i])
        pairx2.append(x2_train[i])
        # print('test',x1_train[i].shape)
        inc = random.randrange(1, num_size)
        dn = (i + inc) % num_size
        pairx1.append(x1_train[dn])
        pairx2.append(x2_train[dn])
        # pairx1n.append(x1_train[dn])
        # pairx2n.append(x2_train[dn])
        # pairx1+=pairx1n
        # pairx2+=pairx2n
        labels+=[1,0]
    return np.array(pairx1),np.array(pairx2),np.array(labels)

    # n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    # print(n,num_classes)
    # for d in range(num_classes):
    #     for i in range(n):
    #         z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
    #         pairs += [[x[z1], x[z2]]]
    #         inc = random.randrange(1, num_classes)
    #         dn = (d + inc) % num_classes
    #         z1, z2 = digit_indices[d][i], digit_indices[dn][i]
    #         pairs += [[x[z1], x[z2]]]
    #         labels += [1, 0]
    # return np.array(pairs), np.array(labels)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    print("y_true",y_true)
    print("y_pred",y_pred)
    print("pred",pred)
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def compute_recall(model):
    correct=0
    for i in range(no_samples):
        # print(i)
        qF=X2_test[i]
        qF =np.tile(qF,(no_samples,1))
        imF=X1_test
        sc=model.predict([imF,qF])
        sc=sc.ravel()
        r=sc.argsort()[:5]
        print(r,i)
        # print(sc)
        if i in r:
            correct+=1

    recall_5=correct/no_samples
    return recall_5
# the data, shuffled and split between train and test sets
# X1_train = np.load('train.npy')
# X2_train = np.load('X2Train.npy')
# X1_test = np.load('X1_test_short.npy')
# X2_test = np.load('X2_test_short.npy')
# tr_pairs, tr_y = create_pairs(x1_train,x2_train)
# no_samples = X1_test.shape[0]
# print("shape",X1_test.shape[1])
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# input_dim = 784
epochs = 30

# create training+test positive and negative pairs
# digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
# print('digit_indices',digit_indices)
# tr_pairs, tr_y = create_pairs(x_train, digit_indices)
# print('pairs,labels shape',tr_pairs.shape,tr_y.shape)
# print(tr_pairs,tr_y)
# digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
# te_pairs, te_y = create_pairs(X1_test, X2_test)
# te_pairs = np.matrix(te_pairs)
# te_pairs[:,0]=np.matrix(te_pairs[:,0])
tr1,tr2,tr_y = np.load('tr1.npy'), np.load('tr2.npy'),np.load('tr_y.npy')
# te1,te2,tey =np.load('tr1.npy'),np.load('tr2.npy'),np.load('tr_y.npy')
# 2
# np.save('tr1.npy',tr1)
# np.save('tr2.npy',tr2)
# np.save('tr_y.npy',tr_y)
# np.save('te1.npy',te1)
# np.save('te2.npy',te2)
# np.save('tey.npy',tey)

# print('pairs,labels shape',tp1.shape,tp2.shape,ty.shape)

# network definition
# base_network = create_base_network(input_dim)

input_a = Input(shape=(29188,))


input_b = Input(shape=(4800,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
project_a = Input(shape=(29188,))
x = Dense(4800,activation ='relu')(project_a)


shared_fc = Dense(128)
processed_a = shared_fc(x)
processed_b = shared_fc(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([ project_a,input_b], [distance])

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
# model.load_weights("/home/subha/RefExpCVPR2018/penseur/finalTrain/modelFinalnew.h5")
history = model.fit([tr1, tr2], tr_y,
          batch_size=30,
          epochs=epochs,
          validation_split = 0.2)
# K.clear_session()
model.save_weights("/home/subha/RefExpCVPR2018/penseur/finalTrain/modelFinalnew.h5")
model_json = model.to_json()
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('lossFinal.jpg')
# plt.show()
with open("/home/subha/RefExpCVPR2018/penseur/finalTrain/modelFinal.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
#
# print("Saved model to disk")
# plot_model(model, to_file='modelFinal.png',show_shapes=True)
# compute final accuracy on training and test sets
# recall_5=compute_recall(model)
#
# y_pred = model.predict([tr1, tr2])
# tr_acc = compute_accuracy(tr_y, y_pred)
# y_pred = model.predict([te1,te2])
# te_acc = compute_accuracy(tey, y_pred)
# # results=[tr_acc,te_acc,recall_5]
# # thefile = open('results_train.txt', 'w')
# # for item in results:
# #   thefile.write("%s\n" % item)
# # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
# print('Recall@5: %0.2f%%'%(recall_5))
