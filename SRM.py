import tensorflow as tf

def SRM_block(x, channels, use_bias=False, is_training=True, scope='srm_block'):
    with tf.variable_scope(scope) :
        bs, h, w, c = x.get_shape().as_list() # c = channels

        x = tf.reshape(x, shape=[bs, -1, c]) # [bs, h*w, c]

        x_mean, x_var = tf.nn.moments(x, axes=1, keep_dims=True) # [bs, 1, c]
        x_std = tf.sqrt(x_var + 1e-5)

        t = tf.concat([x_mean, x_std], axis=1) # [bs, 2, c]

        z = tf.layers.conv1d(t, channels, kernel_size=2, strides=1, use_bias=use_bias)
        z = tf.layers.batch_normalization(z, momentum=0.9, epsilon=1e-05, center=True, scale=True, training=is_training, name=scope)
        # z = tf.contrib.layers.batch_norm(z, decay=0.9, epsilon=1e-05, center=True, scale=True, updates_collections=None, is_training=is_training, scope=scope)

        g = tf.sigmoid(z)

        x = tf.reshape(x * g, shape=[bs, h, w, c])

        return x
