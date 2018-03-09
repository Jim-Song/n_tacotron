import argparse
from datetime import datetime
import math
import numpy as np
import os
import subprocess
import time
import tensorflow as tf
import traceback

from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, infolog, plot, ValueWindow
from weight_norm_fbank_10782_20171124_low25_high3700_global_norm.ResCNN_wn_A_softmax_prelu_2deeper_group import ResCNN
log = infolog.log


def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit



def add_stats(model, gradients, learning_rate):
  with tf.variable_scope('stats') as scope:
    tf.summary.histogram('linear_outputs', model.linear_outputs)
    tf.summary.histogram('linear_targets', model.linear_targets)
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('loss_mel', model.mel_loss)
    tf.summary.scalar('loss_linear', model.linear_loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss', model.loss)

    name_list = [('model/unit_1', 'unit_1'),
                 ('model/unit_2', 'unit_2'),
                 ('model/unit_last', 'unit_last'),
                 ('model/unit_ln', 'unit_ln'),
                 ('model/inference/embedding', 'embedding'),
                 ('model/inference/prenet', 'encoder_prenet'),
                 ('model/inference/encoder_cbhg', 'encoder_cbhg'),
                 ('model/inference/decoder/output_projection_wrapper/multi_rnn_cell/cell_0/output_projection_wrapper/concat_output_and_attention_wrapper/attention_wrapper/decoder_prenet_wrapper/decoder_prenet', 'decoder_prenet'),
                 ('model/inference/decoder/output_projection_wrapper/multi_rnn_cell/cell_0/output_projection_wrapper/concat_output_and_attention_wrapper/attention_wrapper/decoder_prenet_wrapper/gru_cell', 'gru_cell'),
                 ('model/inference/decoder/output_projection_wrapper/multi_rnn_cell/cell_0/output_projection_wrapper/concat_output_and_attention_wrapper/attention_wrapper/bahdanau_attention', 'bahdanau_attention'),
                 ('model/inference/decoder/output_projection_wrapper/multi_rnn_cell/cell_0/output_projection_wrapper/kernel', 'kernel1'),
                 ('model/inference/decoder/output_projection_wrapper/multi_rnn_cell/cell_0/output_projection_wrapper/bias', 'bias1'),
                 ('model/inference/decoder/output_projection_wrapper/multi_rnn_cell/cell_1', 'cell1'),
                 ('model/inference/decoder/output_projection_wrapper/multi_rnn_cell/cell_2', 'cell2'),
                 ('model/inference/decoder/output_projection_wrapper/kernel', 'kernel2'),
                 ('model/inference/decoder/output_projection_wrapper/bias', 'bias2'),
                 ('model/inference/post_cbhg', 'post_cbhg')]

    gradient_norms = []
    for _ in name_list:
      gradient_norms.append([])
    gradient_norms.append([])

    for grad in gradients:
      in_name_list = False
      #print(grad[1].op.name)
      #print('----------------------------------------------------------------------')
      for index, name in enumerate(name_list):
        if grad[1].op.name.find(name[0]) > -1:
          #print(index)
          in_name_list = True
          gradient_norms[index].append(tf.norm(grad))
      if not in_name_list:
        gradient_norms[-1].append(tf.norm(grad))

    for index, name in enumerate(name_list):
      gradient_norm = gradient_norms[index]
      tf.summary.histogram('gradient_norm_' + name[1], gradient_norm)
      tf.summary.scalar('max_gradient_norm_' + name[1], tf.reduce_max(gradient_norm))
      tf.summary.scalar('sum_gradient_norm_' + name[1], tf.reduce_sum(gradient_norm))

    tf.summary.histogram('gradient_norm' + '_not_in_list', gradient_norms[-1])
    tf.summary.scalar('max_gradient_norm' + '_not_in_list', tf.reduce_max(gradient_norms[-1]))
    tf.summary.scalar('sum_gradient_norm' + '_not_in_list', tf.reduce_sum(gradient_norms[-1]))

    gradient_norms2 = [tf.norm(grad) for grad in gradients]
    gradient_norms_square = [(tf.norm(grad))**2 for grad in gradients]
    tf.summary.histogram('gradient_norm', gradient_norms2)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms2))
    tf.summary.scalar('sum_gradient_norm', tf.reduce_sum(gradient_norms2))
    tf.summary.scalar('sum_square_gradient_norm', tf.reduce_sum(gradient_norms_square))
    return tf.summary.merge_all()



def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')

# ---------------------------------------------------------------------------------
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def _learning_rate_decay(init_lr, global_step, num_gpu=1):
    # Noam scheme from tensor2tensor:
    warmup_steps = 20000.0
    step = tf.cast(global_step * (num_gpu + 1) / 2 + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.4 * tf.minimum(step * warmup_steps ** -1.4, step ** -0.4) * (num_gpu + 1) / 2
# ---------------------------------------------------------------------------------



def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  input_path = os.path.join(args.base_dir, args.input)
  log('Checkpoint path: %s' % checkpoint_path)
  log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  # graph
  with tf.Graph().as_default(), tf.device('/cpu:0'):

    #new attributes of hparams
    #hparams.num_GPU = len(GPUs_id)
    #hparams.datasets = eval(args.datasets)
    hparams.datasets = eval(args.datasets)
    hparams.prenet_layer1 = args.prenet_layer1
    hparams.prenet_layer2 = args.prenet_layer2
    hparams.gru_size = args.gru_size
    hparams.attention_size = args.attention_size
    hparams.rnn_size = args.rnn_size
    hparams.enable_fv1 = args.enable_fv1
    hparams.enable_fv2 = args.enable_fv2


    if args.batch_size:
      hparams.batch_size = args.batch_size

    # Multi-GPU settings
    GPUs_id = eval(args.GPUs_id)
    hparams.num_GPU = len(GPUs_id)
    tower_grads = []
    tower_loss = []
    models = []

    global_step = tf.Variable(-1, name='global_step', trainable=False)
    if hparams.decay_learning_rate:
      learning_rate = _learning_rate_decay(hparams.initial_learning_rate, global_step, hparams.num_GPU)
    else:
      learning_rate = tf.convert_to_tensor(hparams.initial_learning_rate)
    # Set up DataFeeder:
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
      input_path = os.path.join(args.base_dir, args.input)
      feeder = DataFeeder(coord, input_path, hparams)
      inputs = feeder.inputs
      inputs = tf.split(inputs, hparams.num_GPU, 0)
      input_lengths = feeder.input_lengths
      input_lengths = tf.split(input_lengths, hparams.num_GPU, 0)
      mel_targets = feeder.mel_targets
      mel_targets = tf.split(mel_targets, hparams.num_GPU, 0)
      linear_targets = feeder.linear_targets
      linear_targets = tf.split(linear_targets, hparams.num_GPU, 0)

    # Set up model:
    with tf.variable_scope('model') as scope:
      optimizer = tf.train.AdamOptimizer(learning_rate, hparams.adam_beta1, hparams.adam_beta2)
      for i, GPU_id in enumerate(GPUs_id):
        with tf.device('/gpu:%d' % GPU_id):
          with tf.name_scope('GPU_%d' % GPU_id):

            if hparams.enable_fv1 or hparams.enable_fv2:
              net = ResCNN(data=mel_targets[i], batch_size=hparams.batch_size, hyparam=hparams)
              net.inference()

              voice_print_feature = tf.reduce_mean(net.features, 0)
            else:
              voice_print_feature = None


            models.append(None)
            models[i] = create_model(args.model, hparams)
            models[i].initialize(inputs=inputs[i], input_lengths=input_lengths[i],
                                 mel_targets=mel_targets[i], linear_targets=linear_targets[i],
                                 voice_print_feature=voice_print_feature)
            models[i].add_loss()


            """L2 weight decay loss."""
            if args.weight_decay > 0:
              costs = []
              for var in tf.trainable_variables():
                #if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                  # tf.summary.histogram(var.op.name, var)
              weight_decay = tf.cast(args.weight_decay, tf.float32)
              cost = models[i].loss
              models[i].loss += tf.multiply(weight_decay, tf.add_n(costs))
              cost_pure_wd = tf.multiply(weight_decay, tf.add_n(costs))
            else:
              cost = models[i].loss
              cost_pure_wd = tf.constant([0])


            tower_loss.append(models[i].loss)

            tf.get_variable_scope().reuse_variables()
            models[i].add_optimizer(global_step, optimizer)

            tower_grads.append(models[i].gradients)

      # calculate average gradient
      gradients = average_gradients(tower_grads)

      stats = add_stats(models[0], gradients, learning_rate)
      time.sleep(10)

    # apply average gradient

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      apply_gradient_op = optimizer.apply_gradients(gradients, global_step=global_step)


    # Bookkeeping:
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    # Train!
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      try:
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        if args.restore_step:
          # Restore from a checkpoint if the user requested it.
          restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
          saver.restore(sess, restore_path)
          log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
        else:
          log('Starting new training run at commit: %s' % commit, slack=True)

        feeder.start_in_session(sess)

        while not coord.should_stop():
          start_time = time.time()
          model = models[0]

          step, loss, opt , loss_wd, loss_pure_wd = sess.run([global_step, cost, apply_gradient_op, model.loss, cost_pure_wd])
          feeder._batch_in_queue -= 1
          log('feed._batch_in_queue: %s' % str(feeder._batch_in_queue), slack=True)

          time_window.append(time.time() - start_time)
          loss_window.append(loss)
          message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, loss_wd=%.05f, loss_pure_wd=%.05f]' % (
            step, time_window.average, loss, loss_window.average, loss_wd, loss_pure_wd)
          log(message, slack=(step % args.checkpoint_interval == 0))

          #if the gradient seems to explode, then restore to the previous step
          if loss > 2 * loss_window.average or math.isnan(loss):
            log('recover to the previous checkpoint')
            #tf.reset_default_graph()
            restore_step = int((step - 10) / args.checkpoint_interval) * args.checkpoint_interval
            restore_path = '%s-%d' % (checkpoint_path, restore_step)
            saver.restore(sess, restore_path)
            continue


          if loss > 100 or math.isnan(loss):
            log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
            raise Exception('Loss Exploded')

          try:
            if step % args.summary_interval == 0:
              log('Writing summary at step: %d' % step)
              summary_writer.add_summary(sess.run(stats), step)
          except:
            pass

          if step % args.checkpoint_interval == 0:
            log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
            saver.save(sess, checkpoint_path, global_step=step)
            log('Saving audio and alignment...')
            input_seq, spectrogram, alignment = sess.run([
              model.inputs[0], model.linear_outputs[0], model.alignments[0]])
            waveform = audio.inv_spectrogram(spectrogram.T)
            audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
            plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
              info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))
            log('Input: %s' % sequence_to_text(input_seq))

      except Exception as e:
        log('Exiting due to exception: %s' % e, slack=True)
        traceback.print_exc()
        coord.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('./'))
  parser.add_argument('--input', default='training')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  parser.add_argument('--GPUs_id', default='[0]', help='The GPUs\' id list that will be used. Default is 0')
  parser.add_argument('--description', default=None, help='description of the model')
  parser.add_argument('--datasets', default="['npy_ljspeech','npy_vctk']", help='the datasets used for training')# "['npy_vctk', 'npy_ljspeech']"
  parser.add_argument('--batch_size', default=None, type=int, help='batch_size')  #
  parser.add_argument('--prenet_layer1', default=256, type=int, help='prenet_layer1')  #
  parser.add_argument('--prenet_layer2', default=128, type=int, help='prenet_layer2')  #
  parser.add_argument('--gru_size', default=256, type=int, help='gru_size')  #
  parser.add_argument('--attention_size', default=256, type=int, help='attention_size')  #
  parser.add_argument('--rnn_size', default=256, type=int, help='rnn_size')  #
  parser.add_argument('--weight_decay', default=0, type=float, help='weight_decay')  #
  parser.add_argument('--enable_fv1', default=True, type=bool, help='enable_fv1')  #
  parser.add_argument('--enable_fv2', default=True, type=bool, help='enable_fv2')  #


  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s-%s' % (run_name, args.description))
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  train(log_dir, args)


if __name__ == '__main__':
  main()
