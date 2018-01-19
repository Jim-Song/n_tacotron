import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages


class ResCNN(object):
  """ResNet model."""

  #def __init__(self, data, labels, num_classes, batch_size, weight_decay, bottleneck, mode, cos_m, lamda=None):
  def __init__(self, data, labels=None, num_classes=None, batch_size=32, weight_decay=None, bottleneck=True, mode='train', cos_m=1, lamda=None, hyparam=None):
    """ResNet constructor.
    Args:
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.data2 = data
    self.data = tf.expand_dims(self.data2, -1)

    self.num_classes = num_classes
    if labels is not None:
      self.labels = tf.reshape(tf.one_hot(labels, tf.constant(num_classes)),[batch_size, num_classes])
    self.mode = mode
    self.batch_size = tf.shape(data)[0]
    self.weight_decay = weight_decay
    self._extra_train_ops = []
    self.use_bottleneck = bottleneck
    self.init = False
    self.lamda = lamda
    self.cos_m = cos_m
    self._hparams = hyparam

  def inference(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):

      x = self.data

    activate_before_residual = [True, False, False]
    if self.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [64, 128, 256, 512]
    else:
      res_func = self._residual
      filters = [64, 128, 256, 512]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      # filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 4
    if self.cos_m == 1:
      angular_func = self._angular_fully_connected_m1
    elif self.cos_m == 2:
      angular_func = self._angular_fully_connected_m2
    elif self.cos_m == 3:
      angular_func = self._angular_fully_connected_m3
    elif self.cos_m == 4:
      angular_func = self._angular_fully_connected_m4
    else:
      print('--------------cos_m: %d has not implemented'%self.cos_m)
      return

    with tf.variable_scope('unit_1'):
      x = self._conv_wn('unit_1_conv', x, 5, 1, filters[0], self._stride_arr(2))
    for i in six.moves.range(1, 6):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[0], filters[0], self._stride_arr(1), False)

    with tf.variable_scope('unit_2'):
      x = self._conv_wn('unit_2_conv', x, 5, filters[0], filters[1], self._stride_arr(2))
    for i in six.moves.range(1, 6):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    # with tf.variable_scope('unit_3'):
      # x = self._conv_wn('unit_3_conv', x, 5, filters[1], filters[2], self._stride_arr(2))
    # for i in six.moves.range(1, 3):
      # with tf.variable_scope('unit_3_%d' % i):
        # x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    # with tf.variable_scope('unit_4'):
      # x = self._conv_wn('unit_4_conv', x, 5, filters[2], filters[3], self._stride_arr(2))
    # for i in six.moves.range(1, 3):
      # with tf.variable_scope('unit_4_%d' % i):
        # x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._prelu(x)
      x = self._global_avg_pool(x)

    with tf.variable_scope('unit_ln'):
      x = tf.reshape(x, [self.batch_size, int(2048/64*self._hparams.num_mels)])
      #print(x.get_shape())
      #print('1')
      x = self._fully_connected_wn(x, self._hparams.voice_print_size)

    #with tf.variable_scope('logit'):

      # ---------------------------------------------
      #logits = angular_func(x, self.num_classes)
      # ---------------------------------------------

      #self.logits = logits

      # ---------------------------------------------
      #self.predictions = tf.nn.softmax(logits)
      # ---------------------------------------------
    
    x= tf.nn.l2_normalize(x, dim=1)
    self.features = x

  def original_loss(self):
    with tf.variable_scope('original_costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
        logits=self.original_logits, labels=self.labels)
      self.original_cost = tf.reduce_mean(xent, name='xent')
      self.original_cost += self._decay()
      
    tf.add_to_collection('original_losses', self.original_cost)
    tf.summary.scalar('original_cost', self.original_cost)
    
  def loss(self):
    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

    tf.add_to_collection('losses', self.cost)
    tf.summary.scalar('cost', self.cost)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._prelu(x)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._prelu(x)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._prelu(x)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        # x = self._batch_norm('init_bn', x)
        x = self._prelu(x)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        # x = self._batch_norm('init_bn', x)
        x = self._prelu(x)

    with tf.variable_scope('sub1'):
      x = self._conv_wn('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
      # x = self._batch_norm('bn2', x)
      x = self._prelu(x)
      x = self._conv_wn('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      # x = self._batch_norm('bn3', x)
      x = self._prelu(x)
      x = self._conv_wn('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv_wn('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)
    self.weight_decay = tf.cast(self.weight_decay, tf.float32)
    return tf.multiply(self.weight_decay, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')
      
  def _conv_wn(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    out_filters = int(out_filters)
    with tf.variable_scope(name):
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
      # kernel_norm = tf.nn.l2_normalize(kernel.initialized_value(), [0, 1, 2])
      g = tf.get_variable('g', [out_filters], dtype=tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32), trainable=True)
      b = tf.get_variable('b', [out_filters], dtype=tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32), trainable=True)
      W = tf.reshape(g, [1, 1, 1, out_filters]) * \
            tf.nn.l2_normalize(kernel, [0, 1, 2])
      x = tf.nn.bias_add(tf.nn.conv2d(x, W, strides, padding='SAME'), b)
      return x

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    
  def _prelu(self, x):
    alphas = tf.get_variable(
        'alpha', x.get_shape()[-1], dtype=tf.float32, 
        initializer=tf.constant_initializer(0.0))
    new_shape = x.get_shape().as_list()
    for i in range(len(new_shape)-1):
      new_shape[i] = 1
    new_shape[-1] = -1
    new_shape = np.array(new_shape)
    x = tf.maximum(x,0)+tf.reshape(alphas,new_shape)*tf.minimum(x,0)
    return x
    
  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],tf.float32,
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],tf.float32,
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)
    
  def _fully_connected_wn(self, x, out_dim, init_scale=1.):
    #x = tf.reshape(x, [self.batch_size, -1])
    #print(x.get_shape())
    w = tf.get_variable('DW', [x.get_shape()[1], self._hparams.voice_print_size], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
    g = tf.get_variable('g', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=True)
    b = tf.get_variable('b', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=True)
    x = tf.matmul(x, w)
    scaler = g / tf.sqrt(tf.reduce_sum(tf.square(w), [0]))
    x = tf.reshape(scaler, [1, out_dim]) * x + tf.reshape(b, [1, out_dim])
    return x
      
  def _angular_fully_connected_m4(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],tf.float32,
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    w = tf.nn.l2_normalize(w, dim = 0)
    x_norm = tf.norm(x, ord=2, axis = 1)
    xw = tf.matmul(x, w)
    cos_theta = xw / tf.reshape(x_norm, [-1, 1])
    cos_theta_quadratic = tf.multiply(cos_theta, cos_theta)
    cos_theta_quartic = tf.multiply(cos_theta_quadratic, cos_theta_quadratic)
    sign_0 = tf.sign(cos_theta)
    sign_3 = tf.multiply(sign_0, tf.sign(2*cos_theta_quadratic-1))
    sign_4 = 2*sign_0 + sign_3 - 3
    mask = tf.equal(self.labels, 1.0)
    xw_m = 8.0*cos_theta_quartic - 8.0*cos_theta_quadratic + 1
    xw_m = tf.multiply(sign_3, xw_m) + sign_4
    xw_m = tf.reshape(x_norm,[-1,1])* xw_m
    x = tf.where(mask, xw_m, xw)
    x = (x + self.lamda*xw)/(1 + self.lamda)
    # no lamda angular
    self.original_logits = xw
    self.original_predictions = tf.nn.softmax(self.original_logits)
    return x
    
  def _angular_fully_connected_m3(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],tf.float32,
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    # # no angular
    # self.original_logits = tf.matmul(x, w)
    # self.original_predictions = tf.nn.softmax(self.original_logits)
    # angular
    w = tf.nn.l2_normalize(w, dim = 0)
    x_norm = tf.norm(x, ord=2, axis = 1)
    xw = tf.matmul(x, w)
    cos_theta = xw / tf.reshape(x_norm, [-1, 1])
    cos_theta_quadratic = tf.multiply(cos_theta, cos_theta)
    cos_theta_cubic = tf.multiply(cos_theta_quadratic, cos_theta)
    sign_0 = tf.sign(cos_theta)
    sign_1 = tf.sign(tf.abs(cos_theta) - 0.5)
    sign_2 = sign_0 * (1 + sign_1) -2

    mask = tf.equal(self.labels, 1.0)
    xw_m = sign_1*(4*cos_theta_cubic-3*cos_theta)+sign_2
    xw_m = tf.reshape(x_norm,[-1,1])* xw_m
    x = tf.where(mask, xw_m, xw)
    x = (x + self.lamda*xw)/(1 + self.lamda)
    # no lamda angular
    self.original_logits = xw
    self.original_predictions = tf.nn.softmax(self.original_logits)
    return x
    
  def _angular_fully_connected_m2(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],tf.float32,
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    w = tf.nn.l2_normalize(w, dim = 0)
    x_norm = tf.norm(x, ord=2, axis = 1)
    xw = tf.matmul(x, w)
    cos_theta = xw / tf.reshape(x_norm, [-1, 1])
    cos_theta_quadratic = tf.multiply(cos_theta, cos_theta)
    sign_0 = tf.sign(cos_theta)
    mask = tf.equal(self.labels, 1.0)
    xw_m = 2*sign_0*cos_theta_quadratic - 1
    xw_m = tf.reshape(x_norm,[-1,1])* xw_m
    x = tf.where(mask, xw_m, xw)
    x = (x + self.lamda*xw)/(1 + self.lamda)
    # no lamda angular
    self.original_logits = xw
    self.original_predictions = tf.nn.softmax(self.original_logits)
    return x
    
  def _angular_fully_connected_m1(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],tf.float32,
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    w = tf.nn.l2_normalize(w, dim = 0)
    x = tf.matmul(x, w)
    # no lamda angular
    self.original_logits = x
    self.original_predictions = tf.nn.softmax(self.original_logits)
    return x

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, 1)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]
    
  def _get_var_maybe_avg(self, var_name):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name)
    # if ema is not None:
        # v = ema.average(v)
    return v

  def _get_vars_maybe_avg(self, var_names):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(self._get_var_maybe_avg(vn))
    return vars