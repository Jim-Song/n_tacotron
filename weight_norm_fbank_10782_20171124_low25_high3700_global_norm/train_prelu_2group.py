import tensorflow as tf
import os.path
import re
import time
from ResCNN_wn_A_softmax_prelu_2deeper_group import ResCNN
import numpy as np

from datetime import datetime
#from six.moves import xrange  # pylint: disable=redefined-builtin

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './logs_2deep_group',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('rec_name', '/hdd1/audio_data/tfrecord_single_axis_all_combined_data_freq25_3700_10782ps.tfrecord',
                           """rec file directory""")
tf.app.flags.DEFINE_string('mode', 'train',
                           """train or eval""")
tf.app.flags.DEFINE_float('lr', 0.001,
                            """the initial learning rate.""")
tf.app.flags.DEFINE_float('wd', 0.0005,
                            """the initial learning rate.""")
tf.app.flags.DEFINE_float('decay_factor', 0.01,
                            """the decay_factor of learning rate.""")
tf.app.flags.DEFINE_float('moving_average_decay', 1.0,
                            """the decay_factor of learning rate.""")
tf.app.flags.DEFINE_integer('num_epoch_per_decay', 25,
                            """num of epochs to decay.""")
tf.app.flags.DEFINE_integer('steps_per_saving', 10000,
                            """steps per saving model""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Size of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps', 300000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_classes', 10782,
                            """Number of examples to run per epoch.""")
tf.app.flags.DEFINE_integer('num_examples', 1060831,
                            """Number of examples to run per epoch.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_string('gpus', '0,1',
                            """set CUDA_VISIBLE_DEVICES""")
tf.app.flags.DEFINE_boolean('bt_neck', True,
                            """Whether to use bottleneck.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('retrain', False,
                            """Whtether to retrain """)
tf.app.flags.DEFINE_string('restore_path', './logs_2deep_group/model.ckpt-180000',
                            """the path of restoring model""")
tf.app.flags.DEFINE_integer('cos_m', 3,
                            """constraits on angular margin""")
tf.app.flags.DEFINE_float('lamda_base', 20,
                            """base for calculating lamda""")
tf.app.flags.DEFINE_float('lamda_power', 1,
                            """power for calculating lamda""")
tf.app.flags.DEFINE_float('lamda_min', 4,
                            """the low bound of lamda""")
tf.app.flags.DEFINE_float('lamda_gama', 0,
                            """gama for calculating lamda""")
tf.app.flags.DEFINE_float('lamda', 0,
                            """lamda for calculating A-softmax""")
tf.app.flags.DEFINE_float('lamda_init', 20.0,
                            """base for calculating lamda""")
tf.app.flags.DEFINE_float('lamda_decay_factor', 0.02,
                            """power for calculating lamda""")
tf.app.flags.DEFINE_integer('lamda_num_epoch_per_decay', 90,
                            """gama for calculating lamda""")

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
                            

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'data': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64)
      }
  )
  data = tf.decode_raw(features['data'], tf.float64)
  data = tf.cast(data, tf.float32)
  data = tf.reshape(data, [500,64,1])

  label = tf.cast(features['label'], tf.int64)

  return data, label

def input(filename, batch_size):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])
    image,label = read_and_decode(filename_queue)
    #print image._shape, label._shape
    images,labels = tf.train.batch([image,label],batch_size = batch_size,num_threads=16,
                                             capacity=10+3*batch_size)#, min_after_dequeue=10)
  return images,labels

def tower_loss(scope):
  data, labels = input(FLAGS.rec_name, FLAGS.batch_size)
  net = ResCNN(data, labels, FLAGS.num_classes, FLAGS.batch_size, FLAGS.wd, FLAGS.bt_neck, FLAGS.mode, FLAGS.cos_m, FLAGS.lamda)
  net.inference(False)
  net.loss()
  net.original_loss()
  truth_ = net.labels
  predictions_ = net.predictions
  org_predictions = net.original_predictions
  ret_lamda = net.lamda

  #acc = accuracy(predictions_,truth_)
  predicts = tf.argmax(predictions_, axis=1)
  org_predicts = tf.argmax(org_predictions, axis=1)
  truths = tf.argmax(truth_, axis=1)
  acc = tf.reduce_mean(tf.to_float(tf.equal(predicts, truths)))
  org_acc = tf.reduce_mean(tf.to_float(tf.equal(org_predicts, truths)))

  losses = tf.get_collection('losses', scope)
  original_losses = tf.get_collection('original_losses', scope)
  
  total_loss = tf.add_n(losses, name='total_loss')
  total_org_loss = tf.add_n(original_losses, name='total_original_loss')
  for l in losses + [total_loss] + [total_org_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
    tf.summary.scalar(loss_name, l)
  return total_loss, predicts, truths, acc, ret_lamda, total_org_loss, org_acc

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
  
def average_gradients_adm(grads):
    for i in range(1, FLAGS.num_gpus):
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    return grads[0]

def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999, global_step=0):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1 > 0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1 * v + (1. - mom1) * g
            v_hat = v_t / (1. - tf.pow(mom1, t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2 * mg + (1. - mom2) * tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2, t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    updates.append(global_step.assign_add(1))
    return tf.group(*updates)

def train():
  with tf.Graph().as_default(), tf.device('/gpu:0'):
    global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable=False)
    
    num_batches_per_epoch = (FLAGS.num_examples / FLAGS.batch_size/ FLAGS.num_gpus)
    decay_steps = int(num_batches_per_epoch*FLAGS.num_epoch_per_decay)
    #print decay_steps
    # lr = tf.train.exponential_decay(FLAGS.lr, global_step, decay_steps, FLAGS.decay_factor) 
    boundaries = [170000.0, 250000.0]
    values = [0.001, 0.0003, 0.0001]
    lr = tf.train.piecewise_constant(global_step,boundaries,values)
    lamda_decay_steps = int(num_batches_per_epoch*FLAGS.lamda_num_epoch_per_decay)
    FLAGS.lamda = tf.train.exponential_decay(FLAGS.lamda_init, global_step, lamda_decay_steps, FLAGS.lamda_decay_factor)
    FLAGS.lamda = tf.maximum(FLAGS.lamda, FLAGS.lamda_min)
    
    # opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    tower_grads = []
    tower_accs = []
    # #--init
    # datas, labels = input(FLAGS.rec_name, FLAGS.batch_size)
    # net = ResCNN(datas, labels, FLAGS.num_classes, FLAGS.batch_size, FLAGS.wd, FLAGS.bt_neck, FLAGS.mode, FLAGS.cos_m, FLAGS.lamda)
    # net.inference(True)
    # tf.get_variable_scope().reuse_variables()
    
    # all_params = tf.trainable_variables()

    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('tower', 0)) as scope:
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss, predicts, truths, acc, ret_lamda, org_loss, org_acc = tower_loss(scope)
            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()
            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            # Calculate the gradients for the batch of data on this CIFAR tower.
            # grads = opt.compute_gradients(loss)
            all_params = tf.trainable_variables()
            grads=tf.gradients(loss, all_params)
            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
            tower_accs.append(acc)
    
    grads = average_gradients_adm(tower_grads)
    acc_avg = tf.reduce_mean(tower_accs, 0)

    summaries.append(tf.summary.scalar('learning_rate', lr))
    summaries.append(tf.summary.scalar('accuracy', acc_avg))
    summaries.append(tf.summary.scalar('lamda', FLAGS.lamda))
    summaries.append(tf.summary.scalar('orginal_acc',org_acc))

    for grad, var in zip(grads, all_params):
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    apply_gradient_op=adam_updates(
        all_params, grads, lr=lr, mom1=0.95, mom2=0.9995, global_step=global_step)
    # variable_averages = tf.train.ExponentialMovingAverage(
        # FLAGS.moving_average_decay, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # train_op = tf.group(apply_gradient_op, variables_averages_op)
    
    train_op = tf.group(apply_gradient_op)
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=60)
    restore_variables = tf.global_variables()

    if FLAGS.retrain == True:
        restore_saver = tf.train.Saver(restore_variables)

    summary_op = tf.summary.merge(summaries)

    config = tf.ConfigProto(allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth=True
    config.gpu_options.allocator_type='BFC'
    sess = tf.Session(config=config)
    
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    
    init = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    sess.run(init)
    sess.run(local_init_op)

    # tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    tf.get_default_graph().finalize()
    
    start_step = 0
    if FLAGS.retrain == True:
        restore_saver.restore(sess, FLAGS.restore_path)
        print '---------------->>Restore model from '+FLAGS.restore_path
        tmpstr = FLAGS.restore_path
        start_step = int(tmpstr.split('-')[-1])
        
    for step in xrange(start_step,FLAGS.max_steps):
      start_time = time.time()
      _, loss_value, acc_value, lr_value, org_loss_value, org_acc_value, step_lamda, tmp_global_step= sess.run([train_op,\
                loss, acc_avg, lr, org_loss, org_acc, ret_lamda, global_step])
      duration = time.time() - start_time

      if step % 100 == 0:
        print tmp_global_step
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.5f, org_loss = %.5f, acc = %.5f, org_acc = %.5f, lr = %.5f, lamda = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,\
                org_loss_value, acc_value, org_acc_value, lr_value, step_lamda,
                examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % FLAGS.steps_per_saving == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        
    coord.request_stop()  
    coord.join(threads)


def main(argv=None):
  train()

if __name__ == '__main__':
  tf.app.run()
