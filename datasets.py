import numpy as np
import os
from PIL import Image
import pickle
import gzip


# def load_usps(data_path='./data/usps'):
#     import os
#     if not os.path.exists(data_path+'/usps_train.jf'):
#         if not os.path.exists(data_path+'/usps_train.jf.gz'):
#             os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
#             os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
#         os.system('gunzip %s/usps_train.jf.gz' % data_path)
#         os.system('gunzip %s/usps_test.jf.gz' % data_path)
#
#     with open(data_path + '/usps_train.jf') as f:
#         data = f.readlines()
#     data = data[1:-1]
#     data = [list(map(float, line.split())) for line in data]
#     data = np.array(data)
#     data_train, labels_train = data[:, 1:], data[:, 0]
#
#     with open(data_path + '/usps_test.jf') as f:
#         data = f.readlines()
#     data = data[1:-1]
#     data = [list(map(float, line.split())) for line in data]
#     data = np.array(data)
#     data_test, labels_test = data[:, 1:], data[:, 0]
#     x_ae = np.concatenate((data_train, data_test)).astype('float64')
#     x_cae = np.concatenate((data_train, data_test)).astype('float32')
#     x_cae /= 2.0
#     x_cae = x_cae.reshape([-1, 16, 16, 1])
#     y = np.concatenate((labels_train, labels_test))
#     print('USPS samples', x_ae.shape, x_cae.shape)
#     return x_ae, x_cae, y
def load_usps(data_path='./data/usps'):
    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float32')
    x /= 2.0
    x = x.reshape([-1, 16, 16, 1])
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y

def load_mnist(data_path='./data/mnist'):
    # the data, shuffled and split between train and test sets
    f = np.load(data_path + '/mnist.npz')
    #from keras.datasets import mnist
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x/255.
    print('MNIST:', x.shape)
    return x, y

def load_usps_original(data_path='./data/usps/original_images'):
    data = np.empty((9298, 1, 16, 16), dtype='float32')
    imgs = os.listdir(data_path)
    num = len(imgs)
    for i in range(num):
        img = Image.open(data_path + '/' + imgs[i])
        arr = np.asarray(img, dtype='float32')
        data[i, :, :, :] = arr
    x_ori = data
    x_ori = x_ori.reshape(-1, 16, 16, 1)
    with open('./data/usps/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open('./data/usps/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]
    y = np.concatenate((labels_train, labels_test))
    return x_ori, y

def load_usps_pad(data_path='./data/usps/padded_images'):
    data = np.empty((9298, 1, 28, 28), dtype='float32')
    imgs = os.listdir(data_path)
    num = len(imgs)
    for i in range(num):
        img = Image.open(data_path + '/' + imgs[i])
        arr = np.asarray(img, dtype='float32')
        data[i, :, :, :] = arr
    x_pad = data
    x_pad = x_pad.reshape(-1, 28, 28, 1)
    with open('./data/usps/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open('./data/usps/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]
    y = np.concatenate((labels_train, labels_test))
    return x_pad, y

def load_mnist_edge(data_path='./data'):
    mnist_edge, y = pickle.load(gzip.open(data_path + '/mnist_train_edge.pickle', 'rb'))
    mnist_edge = mnist_edge.reshape([-1, 28, 28 ,1])
    return mnist_edge, y

def load_mnist_original(data_path='./data'):
    mnist_original, y = pickle.load(gzip.open(data_path + '/mnist_train_original.pickle', 'rb'))
    mnist_original = mnist_original.reshape([-1, 28, 28, 1])
    return mnist_original, y

def load_fashion_mnist(data_path='./data/fashion-mnist'):
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x / 255.
    x = np.divide(x, 255.)
    print('Fashion MNIST samples', x.shape)
    return x, y
