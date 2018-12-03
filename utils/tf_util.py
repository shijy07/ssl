""" Wrapper functions for TensorFlow layers.

Author: Jiaying Shi
Email: shijy07@berkeley.edu
Date: January 2017
"""

import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


def _activation_summary(x, tensor_name):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
        tensor_name: name of x
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_summaries(var, var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    Args:
        var: variable
        var_name: name of var

    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(var_name + '/mean', mean)
        with tf.name_scope(var_name + '/stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(var_name + '/stddev', stddev)
        tf.summary.scalar(var_name + '/max', tf.reduce_max(var))
        tf.summary.scalar(var_name + '/min', tf.reduce_min(var))
        tf.summary.histogram(var_name + '/histogram', var)


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer,
                              dtype=dtype)
        return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        use_xavier: bool, whether to use xavier initializer

    Returns:
        Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None, is_summary=False):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxLxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: int
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        if is_summary:
            _variable_summaries(kernel, scope + '_weights')
        outputs = tf.nn.conv1d(inputs, kernel,
                               stride=stride,
                               padding=padding)

        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        if is_summary:
            _variable_summaries(biases, scope + '_biases')
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        if is_summary:
            _activation_summary(outputs, scope + '_activation')
        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None, is_summary=False):
    """ 2D convolution with non-linear operation.

    Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        if is_summary:
            _variable_summaries(kernel, scope + '_weights')
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        if is_summary:
            _variable_summaries(biases, scope + '_biases')
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        if is_summary:
            _activation_summary(outputs, scope + '_activation')
        return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    """ 2D convolution transpose with non-linear operation.

    Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor

    Note: conv2d(conv2d_transpose(a, num_out, ksize, stride),
                 a.shape[-1], ksize, stride) == a
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels, num_in_channels]
        # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        _variable_summaries(kernel, scope + '_weights')
        stride_h, stride_w = stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height,
                        out_width, num_output_channels]

        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                                         [1, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        _variable_summaries(biases, scope + '_biases')
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        _activation_summary(outputs, scope + '_activation')
        return outputs


def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_summary=False):
    """ 3D convolution with non-linear operation.

    Args:
        inputs: 5-D tensor variable BxDxHxWxC
        num_output_chan
        stride: a list of 3 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        # kernel = _variable_with_weight_decay('weights',
        #                                       shape=kernel_shape,
        #                                       use_xavier=use_xavier,
        #                                       stddev=stddev,
        #                                       wd=weight_decay)
        if use_xavier:
            wtinitializer = tf.contrib.layers.xavier_initializer()
        else:
            wtinitializer = tf.truncated_normal_initializer(stddev=stddev)
        kernel = tf.get_variable('weights', shape=kernel_shape,
                                 initializer=wtinitializer,
                                 dtype=tf.float32)
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.conv3d(inputs, kernel,
                               [1, stride_d, stride_h, stride_w, 1],
                               padding=padding)
        biases = tf.get_variable('biases', [num_output_channels],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        # biases = _variable_on_cpu('biases', [num_output_channels],
        #                           tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weigth_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None,
                    is_summary=False):
    """ Fully connected layer with non-linear operation.

    Args:
        inputs: 2-D tensor BxN
        num_outputs: int

    Returns:
        Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units,
                                                     num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weigth_decay)
        if is_summary:
            _variable_summaries(weights, scope + '_weights')
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        if is_summary:
            _variable_summaries(biases, scope + '_weights')
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        if is_summary:
            _activation_summary(outputs, scope + '_activation')
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID',
               is_summary=False):
    """ 2D max pooling.

    Args:
        inputs: 4-D tensor BxHxWxC
        kernel_size: a list of 2 ints
        stride: a list of 2 ints

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        if is_summary:
            _activation_summary(outputs, scope + '_activation')
        return outputs


def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID',
               is_summary=False):
    """ 2D avg pooling.

    Args:
        inputs: 4-D tensor BxHxWxC
        kernel_size: a list of 2 ints
        stride: a list of 2 ints

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        if is_summary:
            _activation_summary(outputs, scope + '_activation')
        return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID',
               is_summary=False):
    """ 3D max pooling.

    Args:
        inputs: 5-D tensor BxDxHxWxC
        kernel_size: a list of 3 ints
        stride: a list of 3 ints

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.max_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h,
                                            stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        if is_summary:
            _activation_summary(outputs, scope + '_activation')
        return outputs


def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID',
               is_summary=False):
    """ 3D avg pooling.

    Args:
        inputs: 5-D tensor BxDxHxWxC
        kernel_size: a list of 3 ints
        stride: a list of 3 ints

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.avg_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h,
                                            stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        if is_summary:
            _activation_summary(outputs, scope + '_activation')
        return outputs


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-
               use-batch-normalization-in-tensorflow

    Args:
            inputs:       Tensor, k-D input ... x C could be BC or
                          BHWC or BDHWC
            is_training:  boolean tf.Varialbe, true indicates training phase
            scope:        string, variable scope
            moments_dims: a list of ints, indicating dimensions for
                          moments calculation
            bn_decay:     float or float tensor variable, controling moving
                          average weight
    Return:
            normed:       batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims,
                                              name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean),
                                     ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean,
                                           var, beta, gamma, 1e-3)
    return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
            inputs:      Tensor, 2D BxC input
            is_training: boolean tf.Varialbe, true indicates training phase
            bn_decay:    float or float tensor variable, controling
                         moving average weight
            scope:       string, variable scope
    Return:
            normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 1D convolutional maps.

    Args:
            inputs:      Tensor, 3D BLC input maps
            is_training: boolean tf.Varialbe, true indicates training phase
            bn_decay:    float or float tensor variable, controling moving
                         average weight
            scope:       string, variable scope
    Return:
            normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope,
                               [0, 1], bn_decay)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.

    Args:
            inputs:      Tensor, 4D BHWC input maps
            is_training: boolean tf.Varialbe, true indicates training phase
            bn_decay:    float or float tensor variable, controling
                         moving average weight
            scope:       string, variable scope
    Return:
            normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope,
                               [0, 1, 2], bn_decay)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 3D convolutional maps.

    Args:
            inputs:      Tensor, 5D BDHWC input maps
            is_training: boolean tf.Varialbe, true indicates training phase
            bn_decay:    float or float tensor variable, controling
                         moving average weight
            scope:       string, variable scope
    Return:
            normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope,
                               [0, 1, 2, 3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

    Args:
        inputs: tensor
        is_training: boolean tf.Variable
        scope: string
        keep_prob: float in [0,1]
        noise_shape: list of ints

    Returns:
        tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(inputs, keep_prob,
                                                noise_shape),
                          lambda: inputs)
        _activation_summary(outputs, scope + '_activation')
        return outputs


def flatten(inputs, scope=None):
    """Flattens the input while maintaining the batch_size.

        Assumes that the first dimension represents the batch.

    Args:
        inputs: a tensor of size [batch_size, ...].
        scope: Optional scope for name_scope.

    Returns:
        a flattened tensor with shape [batch_size, k].
    Raises:
        ValueError: if inputs.shape is wrong.
    """
    if len(inputs.get_shape()) < 2:
        raise ValueError('Inputs must be have a least 2 dimensions')
    dims = inputs.get_shape()[1:]
    k = dims.num_elements()
    with tf.name_scope(scope, 'Flatten', [inputs]):
        return tf.reshape(inputs, [-1, k])


def draw_sample(mu, log_sigma_sq):

    epsilon = tf.random_normal((tf.shape(mu)), 0, 1)
    sample = tf.add(mu, tf.multiply(
        tf.exp(0.5 * log_sigma_sq), epsilon))

    return sample

logc = np.log(2. * np.pi)
c = - 0.5 * np.log(2 * np.pi)


def tf_normal_logpdf(x, mu, log_sigma_sq):

    return (- 0.5 * logc - log_sigma_sq / 2. - tf.div(tf.square(tf.sub(x, mu)), 2 * tf.exp(log_sigma_sq)))


def tf_stdnormal_logpdf(x):

    return (- 0.5 * (logc + tf.square(x)))


def tf_gaussian_ent(log_sigma_sq):

    return (- 0.5 * (logc + 1.0 + log_sigma_sq))


def tf_gaussian_marg(mu, log_sigma_sq):

    return (- 0.5 * (logc + (tf.square(mu) + tf.exp(log_sigma_sq))))


def tf_binary_xentropy(x, y, const=1e-10):

    return - (x * tf.log(tf.clip_by_value(y, const, 1.0)) +
              (1.0 - x) * tf.log(tf.clip_by_value(1.0 - y, const, 1.0)))
