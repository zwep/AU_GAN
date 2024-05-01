import pprint
import tensorflow as tf
from tensorflow import keras
import tf_slim as slim

pp = pprint.PrettyPrinter()

# get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

weight_init = keras.initializers.GlorotUniform()
weight_regularizer = None


def batch_norm(x, name="batch_norm", decay=0.9, scale=True, epsilon=1e-5):
    return keras.layers.BatchNormalization(
        momentum=decay, epsilon=epsilon, scale=scale, name=name
    )(x)


def instance_norm(x_input, name="instance_norm"):
    with tf.compat.v1.variable_scope(name):
        depth = x_input.get_shape()[3]
        scale = tf.compat.v1.get_variable(
            "scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02)
        )  # , dtype=tf.float32
        offset = tf.compat.v1.get_variable(
            "offset", [depth], initializer=tf.constant_initializer(0.0)
        )
        mean, variance = tf.nn.moments(x_input, axes=[1, 2], keepdims=True)
        epsilon = 1e-5
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (x_input - mean) * inv
        return scale * normalized + offset


def conv2d(x_input, output_dim, ks=4, s=2, stddev=0.02, padding="SAME", name="conv2d"):
    with tf.compat.v1.variable_scope(name):
        return slim.conv2d(
            x_input,
            output_dim,
            ks,
            s,
            padding=padding,
            activation_fn=None,
            weights_initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=None,
        )


def deconv2d(x_input, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.compat.v1.variable_scope(name):
        return slim.conv2d_transpose(
            x_input,
            output_dim,
            ks,
            s,
            padding="SAME",
            activation_fn=None,
            weights_initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=None,
        )


def dilated_conv2d(
    x_input, output_dim, ks=3, s=2, stddev=0.02, padding="SAME", name="conv2d"
):
    with tf.compat.v1.variable_scope(name):
        batch, in_height, in_width, in_channels = [int(d) for d in x_input.get_shape()]
        filter = tf.compat.v1.get_variable(
            "filter",
            [ks, ks, in_channels, output_dim],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0, stddev),
        )
        conv = tf.nn.atrous_conv2d(x_input, filter, rate=s, padding=padding, name=name)

        return conv


def one_step(x, ch, kernel, stride, name):
    return lrelu(
        instance_norm(
            conv2d(x, ch, kernel, stride, name=name + "_first_c"), name + "_first_bn"
        )
    )


def one_step_dilated(x, ch, kernel, stride, name):
    return lrelu(
        instance_norm(
            dilated_conv2d(x, ch, kernel, stride, name=name + "_first_c"),
            name + "_first_bn",
        )
    )


def num_steps(x, ch, kernel, stride, num_steps, name):
    for i in range(num_steps):
        x = lrelu(
            instance_norm(
                conv2d(x, ch, kernel, stride, name=name + "_c_" + str(i)),
                name + "_bn_" + str(i),
            )
        )
    return x


def one_step_noins(x, ch, kernel, stride, name):
    return lrelu(conv2d(x, ch, kernel, stride, name=name + "_first_c"))


def num_steps_noins(x, ch, kernel, stride, num_steps, name):
    for i in range(num_steps):
        x = lrelu(conv2d(x, ch, kernel, stride, name=name + "_c_" + str(i)))
    return x


def dis_down(images, kernel_size, stride, n_scale, ch, name):
    backpack = images[0]
    for i in range(n_scale):
        if i == n_scale - 1:
            images[i] = num_steps(
                backpack, ch, kernel_size, stride, n_scale, name + str(i)
            )
        else:
            images[i] = one_step_dilated(
                images[i + 1], ch, kernel_size, 1, name + str(i)
            )
    return images


def dis_down_noins(images, kernel_size, stride, n_scale, ch, name):
    backpack = images[0]
    for i in range(n_scale):
        if i == n_scale - 1:
            images[i] = num_steps_noins(
                backpack, ch, kernel_size, stride, n_scale, name + str(i)
            )
        else:
            images[i] = one_step_noins(images[i + 1], ch, kernel_size, 1, name + str(i))
    return images


def final_conv(images, n_scale, name):
    for i in range(n_scale):
        images[i] = conv2d(images[i], 1, s=1, name=name + str(i))
    return images


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(x_input, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.compat.v1.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable(
            "Matrix",
            [x_input.get_shape()[-1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev),
        )
        bias = tf.compat.v1.get_variable(
            "bias", [output_size], initializer=tf.constant_initializer(bias_start)
        )
        if with_w:
            return tf.matmul(x_input, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x_input, matrix) + bias


def get_ones_like(logit):
    target = []
    for i in range(len(logit)):
        target.append(tf.ones_like(logit[i]))
    return target


def get_zeros_like(logit):
    target = []
    for i in range(len(logit)):
        target.append(tf.zeros_like(logit[i]))
    return target


def conv(
    x,
    channels,
    kernel=4,
    stride=2,
    pad=0,
    pad_type="zero",
    use_bias=True,
    scope="conv_0",
):
    with tf.compat.v1.variable_scope(scope):
        if pad_type == "zero":
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == "reflect":
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT")

        x = keras.layers.conv2d(
            inputs=x,
            filters=channels,
            kernel_size=kernel,
            kernel_initializer=weight_init,
            kernel_regularizer=weight_regularizer,
            strides=stride,
            use_bias=use_bias,
        )

        return x


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return shape


def show_all_variables():
    model_vars = tf.compat.v1.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
