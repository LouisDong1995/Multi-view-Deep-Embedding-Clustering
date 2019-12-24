from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPool2D, UpSampling2D
from keras.engine.topology import Layer, InputSpec
import keras.backend as K
from sklearn.cluster import KMeans
from sklearn import metrics
from keras.utils.vis_utils import plot_model
from time import time
import metrics
from keras.preprocessing.image import ImageDataGenerator


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def multiAE(input1_shape=(28, 28, 1), input2_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    if input1_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    # ConvAE1
    input1 = Input(shape=input1_shape, name='input1')
    conv1_1 = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1_1')(input1)
    conv1_2 = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv1_2')(conv1_1)
    conv1_3 = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv1_3')(conv1_2)
    x = Flatten()(conv1_3)
    embedding1 = Dense(units=filters[3], name='embedding1')(x)
    x = Dense(units=filters[2] * int(input1_shape[0] / 8) * int(input1_shape[0] / 8), activation='relu')(embedding1)
    x = Reshape((int(input1_shape[0] / 8), int(input1_shape[0] / 8), filters[2]))(x)
    deconv1_3 = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv1_3')(x)
    deconv1_2 = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv1_2')(deconv1_3)
    deconv1_1 = Conv2DTranspose(input1_shape[2], 5, strides=2, padding='same', name='deconv1_1')(deconv1_2)

    #ConvAE2
    input2 = Input(shape=input2_shape, name='input2')
    conv2_1 = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv2_1')(input2)
    conv2_2 = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2_2')(conv2_1)
    conv2_3 = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv2_3')(conv2_2)
    x = Flatten()(conv2_3)
    embedding2 = Dense(units=filters[3], name='embedding2')(x)
    x = Dense(units=filters[2] * int(input2_shape[0] / 8) * int(input2_shape[0] / 8), activation='relu')(embedding2)
    x = Reshape((int(input2_shape[0] / 8), int(input2_shape[0] / 8), filters[2]))(x)
    deconv2_3 = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv2_3')(x)
    deconv2_2 = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2_2')(deconv2_3)
    deconv2_1 = Conv2DTranspose(input2_shape[2], 5, strides=2, padding='same', name='deconv2_1')(deconv2_2)


    # # ConvAE3
    # input3 = Input(shape=input3_shape, name='input3')
    # conv3_1 = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv3_1')(input3)
    # conv3_2 = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv3_2')(conv3_1)
    # conv3_3 = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3_3')(conv3_2)
    # x = Flatten()(conv3_3)
    # embedding3 = Dense(units=filters[3], name='embedding3')(x)
    # x = Dense(units=filters[2] * int(input3_shape[0] / 8) * int(input3_shape[0] / 8), activation='relu')(embedding3)
    # x = Reshape((int(input3_shape[0] / 8), int(input3_shape[0] / 8), filters[2]))(x)
    # deconv3_3 = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3_3')(x)
    # deconv3_2 = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv3_2')(deconv3_3)
    # deconv3_1 = Conv2DTranspose(input3_shape[2], 5, strides=2, padding='same', name='deconv3_1')(deconv3_2)
    #
    # # ConvAE4
    # input4 = Input(shape=input4_shape, name='input4')
    # conv4_1 = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv4_1')(input4)
    # conv4_2 = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv4_2')(conv4_1)
    # conv4_3 = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv4_3')(conv4_2)
    # x = Flatten()(conv4_3)
    # embedding4 = Dense(units=filters[3], name='embedding4')(x)
    # x = Dense(units=filters[2] * int(input4_shape[0] / 8) * int(input4_shape[0] / 8), activation='relu')(
    #     embedding4)
    # x = Reshape((int(input4_shape[0] / 8), int(input4_shape[0] / 8), filters[2]))(x)
    # deconv4_3 = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv4_3')(x)
    # deconv4_2 = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv4_2')(
    #     deconv4_3)
    # deconv4_1 = Conv2DTranspose(input4_shape[2], 5, strides=2, padding='same', name='deconv4_1')(deconv4_2)

    embedding = Concatenate(axis=1)([embedding1, embedding2])

    return Model(inputs=[input1, input2], outputs=[deconv1_1, deconv2_1], name='AE'), \
           Model(inputs=[input1, input2], outputs=embedding, name='encoder')


class MDEC(object):
    def __init__(self,
                 input1_shape,
                 input2_shape,
                 # input3_shape,
                 # input4_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256,
                 ):

        super(MDEC, self).__init__()

        #self.dims = dims
        self.n_clusters = n_clusters
        self.input1_shape = input1_shape
        self.input2_shape = input2_shape
        # self.input3_shape = input3_shape
        # self.input4_shape = input4_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        self.multiAE, self.encoder = multiAE(input1_shape, input2_shape, filters)
        self.input1 = self.multiAE.get_layer('input1').output
        self.input2 = self.multiAE.get_layer('input2').output
        # self.input3 = self.multiAE.get_layer('input3').output
        # self.input4 = self.multiAE.get_layer('input4').output
        self.deconv1_1 = self.multiAE.get_layer('deconv1_1').output
        self.deconv2_1 = self.multiAE.get_layer('deconv2_1').output
        # self.deconv3_1 = self.multiAE.get_layer('deconv3_1').output
        # self.deconv4_1 = self.multiAE.get_layer('deconv4_1').output
        self.embedding1 = self.multiAE.get_layer('embedding1').output
        self.embedding2 = self.multiAE.get_layer('embedding2').output
        # self.embedding3 = self.multiAE.get_layer('embedding3').output
        # self.embedding4 = self.multiAE.get_layer('embedding4').output

        # self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)


        # Define MDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.multiAE.input, outputs=[clustering_layer] + self.multiAE.output)

    def pretrain(self, x1, x2, batch_size=256, epochs=500, optimizer='adam', save_dir='results/temp', aug_pretrain=False, alpha=0.001):
        print('...Pretraining...')

        loss1 = alpha * K.sum((K.dot(K.transpose(self.embedding1), self.embedding2))**2)

        self.multiAE.compile(optimizer=optimizer, loss=['mse', 'mse'])
        self.multiAE.add_loss(loss1)
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger(args.save_dir + '/pretrain_log.csv')

        # begin training
        t0 = time()

        self.multiAE.fit([x1, x2], [x1, x2], batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])

        print('Pretraining time: ', time() - t0)
        self.multiAE.save(save_dir + '/pretrain_multiAE_model.h5')
        print('Pretrained weights are saved to %s/pretrain_multiAE_model.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x1, x2):  # extract features from before clustering layer
        return self.encoder.predict([x1, x2])

    def predict(self, x1, x2):
        q, d1, d2, d3, d4 = self.model.predict([x1, x2], verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    # def random_transform(self, x1, x2):
    #     return self.datagen.flow(x1, shuffle=False, batch_size=x1.shape[0]).next(), \
    #            self.datagen.flow(x2, shuffle=False, batch_size=x2.shape[0]).next(), \
    #            self.datagen.flow(x3, shuffle=False, batch_size=x3.shape[0]).next(), \
    #            self.datagen.flow(x4, shuffle=False, batch_size=x4.shape[0]).next()

    def compile(self, loss=['kld', 'mse', 'mse'], loss_weights=[1, 1, 1], optimizer='adam', alpha=0.001):

        loss1 = alpha * K.sum((K.dot(K.transpose(self.embedding1), self.embedding2)) ** 2)

        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
        self.model.add_loss(loss1)

    def fit(self, x1, x2, y=None, batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140, multiAE_weights=None, save_dir='./results/temp', aug_pretrain=False, alpha=0.001):

        print('Update interval', update_interval)
        save_interval = x1.shape[0] / batch_size * 5
        print('Save interval', save_interval)

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and multiAE_weights is None:
            print('...pretraining MDEC using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x1, x2, batch_size, epochs=args.epochs, save_dir=save_dir, aug_pretrain=aug_pretrain, alpha=alpha)
            self.pretrained = True
        elif multiAE_weights is not None:
            self.multiAE.load_weights(multiAE_weights)
            print('multiAE_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict([x1, x2]))
        hidden = self.encoder.predict([x1, x2])
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        acc = metrics.acc(y, self.y_pred)
        x_acc = metrics.acc(y, kmeans.fit_predict(hidden[:, :10]))
        x_rot90_acc = metrics.acc(y, kmeans.fit_predict(hidden[:, 10:20]))
        # x_rot180_acc = metrics.acc(y, kmeans.fit_predict(hidden[:, 20:30]))
        # x_rot270_acc = metrics.acc(y, kmeans.fit_predict(hidden[:, 30:]))
        print('acc = ', acc, 'x_acc = ', x_acc, 'x_rot90_acc = ', x_rot90_acc)
        print('-' * 100)
        prefile = open(save_dir + '/pretrain_final.txt', 'w')
        prefile.write('acc = '+  str(acc) + '\n')
        prefile.write('x_acc = ' + str(x_acc) + '\n')
        prefile.write('x_rot90_acc = ' + str(x_rot90_acc) + '\n')
        # return

        # Step 3: deep clustering
        # logging filek
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/mdec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'L1', 'L2'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _, _= self.model.predict([x1, x2], verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], L1=loss[2], L2=loss[3])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x1.shape[0]:
                # x1_batch, x2_batch, x3_batch, x4_batch = self.random_transform(x1[index * batch_size::], x2[index * batch_size::], x3[index * batch_size::], x4[index * batch_size::])
                loss = self.model.train_on_batch(x=[x1[index * batch_size::],
                                                    x2[index * batch_size::]],
                                                 y=[p[index * batch_size::],
                                                    x1[index * batch_size::],
                                                    x2[index * batch_size::]])
                index = 0
            else:
                # x1_batch, x2_batch, x3_batch, x4_batch = self.random_transform(x1[index * batch_size:(index + 1) * batch_size], x2[index * batch_size:(index + 1) * batch_size], x3[index * batch_size:(index + 1) * batch_size], x4[index * batch_size:(index + 1) * batch_size])
                loss = self.model.train_on_batch(x=[x1[index * batch_size:(index + 1) * batch_size],
                                                    x2[index * batch_size:(index + 1) * batch_size]],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x1[index * batch_size:(index + 1) * batch_size],
                                                    x2[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save MDEC model checkpoints
                print('saving model to:', save_dir + '/mdec_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/mdec_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/mdec_model_final.h5')
        self.model.save_weights(save_dir + '/mdec_model_final.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist',
                        choices=['mnist', 'usps', 'reutersidf10k', 'mnist_test', 'mnist_rot', 'usps_rot', 'fashion_mnist_rot'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--multiAE_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='results/mdec')
    parser.add_argument('--alpha', default=0.001, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--aug_pretrain', action='store_true',
                        help="Whether to use data augmentation during pretraining phase")
    # parser.add_argument('--aug_clustering', action='store_true',
    #                     help="Whether to use data augmentation during clustering phase")
    args = parser.parse_args()
    print(args)

    # load dataset
    # optimizer = SGD(lr=0.1, momentum=0.99)
    from datasets import *

    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
    if args.dataset == 'mnist':  # recommends: n_clusters=10, update_interval=140
        x_ori, y = load_mnist_original()
        x_edg, _ = load_mnist_edge()
    elif args.dataset == 'usps':  # recommends: n_clusters=10, update_interval=30
        #x, y = load_usps('data/usps')
        x_ori, y = load_usps_original()
        x_pad, _ = load_usps_pad()
        update_interval = 30
        pretrain_epochs = 50
    elif args.dataset == 'reutersidf10k':  # recommends: n_clusters=4, update_interval=3
        x, y = load_reuters('data/reuters')
    elif args.dataset == 'mnist_test':
        x, y = load_mnist()
        x, y = x[60000:], y[60000:]
    elif args.dataset == 'usps_rot':
        x, y = load_usps()
        x_rot90 = np.rot90(x, 1, (1, 2))
        x_rot180 = np.rot90(x, 2, (1, 2))
        x_rot270 = np.rot90(x, 3, (1, 2))
    elif args.dataset == 'mnist_rot':
        x, y = load_mnist()
        x_rot90 = np.rot90(x, 1, (1, 2))
        x_rot180 = np.rot90(x, 2, (1, 2))
        x_rot270 = np.rot90(x, 3, (1, 2))
    elif args.dataset == 'fashion_mnist_rot':
        x, y = load_fashion_mnist()
        x_rot90 = np.rot90(x, 1, (1, 2))
        x_rot180 = np.rot90(x, 2, (1, 2))
        x_rot270 = np.rot90(x, 3, (1, 2))

    # prepare the MDEC model
    mdec = MDEC(input1_shape=x.shape[1:], input2_shape=x_rot90.shape[1:], filters=[32, 64, 128, 10], n_clusters=args.n_clusters)
    plot_model(mdec.model, to_file=args.save_dir + '/mdec_model.png', show_shapes=True)
    mdec.model.summary()
    # begin clustering.
    optimizer = 'adam'
    mdec.compile(optimizer=optimizer, alpha=args.alpha)
    mdec.fit(x, x_rot90, y=y, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval,
             save_dir=args.save_dir,
             multiAE_weights=args.multiAE_weights, aug_pretrain=args.aug_pretrain, alpha=args.alpha)
    y_pred = mdec.y_pred
    print(
        'acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))
