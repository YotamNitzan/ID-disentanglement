import tensorflow as tf
import math

num_classes = 85742  # 10572
initializer = 'glorot_normal'
# initializer = tf.keras.initializers.TruncatedNormal(
#     mean=0.0, stddev=0.05, seed=None)
# initializer = tf.keras.initializers.VarianceScaling(
#     scale=0.05, mode='fan_avg', distribution='normal', seed=None)


class Arcfacelayer(tf.keras.layers.Layer):
    def __init__(self, output_dim=num_classes, s=64., m=0.50):
        self.output_dim = output_dim
        self.s = s
        self.m = m
        super(Arcfacelayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1],
                                             self.output_dim),
                                      initializer=initializer,
                                      regularizer=tf.keras.regularizers.l2(
                                          l=5e-4),
                                      trainable=True)
        super(Arcfacelayer, self).build(input_shape)

    def call(self, embedding, labels):
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m  # issue 1
        threshold = math.cos(math.pi - self.m)
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / embedding_norm
        weights_norm = tf.norm(self.kernel, axis=0, keepdims=True)
        weights = self.kernel / weights_norm
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.s * tf.subtract(tf.multiply(cos_t, cos_m),
                                      tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = self.s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=self.output_dim, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(self.s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(
            cos_mt_temp, mask), name='arcface_loss_output')

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
