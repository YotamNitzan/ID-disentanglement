import tensorflow as tf
import os
from model.arcface.arcface import Arcfacelayer

bn_axis = -1
initializer = 'glorot_normal'


def residual_unit_v3(input, num_filter, stride, dim_match, name):
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           #    beta_regularizer=tf.keras.regularizers.l2(
                                           #        l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           name=name + '_bn1')(input)
    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name=name + '_conv1_pad')(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(
                                   l=5e-4),
                               name=name + '_conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           #    beta_regularizer=tf.keras.regularizers.l2(
                                           #        l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           name=name + '_bn2')(x)
    x = tf.keras.layers.PReLU(name=name + '_relu1',
                              alpha_regularizer=tf.keras.regularizers.l2(
                                  l=5e-4))(x)
    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name=name + '_conv2_pad')(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3),
                               strides=stride,
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(
                                   l=5e-4),
                               name=name + '_conv2')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           #    beta_regularizer=tf.keras.regularizers.l2(
                                           #        l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           name=name + '_bn3')(x)
    if (dim_match):
        shortcut = input
    else:
        shortcut = tf.keras.layers.Conv2D(num_filter, (1, 1),
                                          strides=stride,
                                          padding='valid',
                                          kernel_initializer=initializer,
                                          use_bias=False,
                                          kernel_regularizer=tf.keras.regularizers.l2(
                                              l=5e-4),
                                          name=name + '_conv1sc')(input)
        shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                                      scale=True,
                                                      momentum=0.9,
                                                      epsilon=2e-5,
                                                      #   beta_regularizer=tf.keras.regularizers.l2(
                                                      #       l=5e-4),
                                                      gamma_regularizer=tf.keras.regularizers.l2(
                                                          l=5e-4),
                                                      name=name + '_sc')(shortcut)
    return x + shortcut


def get_fc1(input):
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           #    beta_regularizer=tf.keras.regularizers.l2(
                                           #        l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           name='bn1')(input)
    x = tf.keras.layers.Dropout(0.4)(x)
    resnet_shape = input.shape
    x = tf.keras.layers.Reshape(
        [resnet_shape[1] * resnet_shape[2] * resnet_shape[3]], name='reshapelayer')(x)
    x = tf.keras.layers.Dense(512,
                              name='E_DenseLayer', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(
                                  l=5e-4),
                              bias_regularizer=tf.keras.regularizers.l2(
                                  l=5e-4))(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,
                                           scale=False,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           #    beta_regularizer=tf.keras.regularizers.l2(
                                           #        l=5e-4),
                                           name='fc1')(x)
    return x


def ResNet50():

    input_shape = [112, 112, 3]
    filter_list = [64, 64, 128, 256, 512]
    units = [3, 4, 14, 3]
    num_stages = 4

    img_input = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name='conv0_pad')(img_input)
    x = tf.keras.layers.Conv2D(64, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(
                                   l=5e-4),
                               name='conv0')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           #    beta_regularizer=tf.keras.regularizers.l2(
                                           #        l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           name='bn0')(x)
    # x = tf.keras.layers.Activation('prelu')(x)
    x = tf.keras.layers.PReLU(
        name='prelu0',
        alpha_regularizer=tf.keras.regularizers.l2(
            l=5e-4))(x)

    for i in range(num_stages):
        x = residual_unit_v3(x, filter_list[i + 1], (2, 2), False,
                             name='stage%d_unit%d' % (i + 1, 1))
        for j in range(units[i] - 1):
            x = residual_unit_v3(x, filter_list[i + 1], (1, 1),
                                 True, name='stage%d_unit%d' % (i + 1, j + 2))

    x = get_fc1(x)

    # Create model.
    model = tf.keras.models.Model(img_input, x, name='resnet50')
    model.trainable = True
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
        # if ('conv0' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('bn0' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('prelu0' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage1' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage2' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage3' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage4' in model.layers[i].name):
        #     model.layers[i].trainable = False

    return model


class train_model(tf.keras.Model):
    def __init__(self):
        super(train_model, self).__init__()
        self.resnet = ResNet50()
        self.arcface = Arcfacelayer()

    def call(self, x, y):
        x = self.resnet(x)
        return self.arcface(x, y)
