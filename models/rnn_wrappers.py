import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from .modules import prenet


class DecoderPrenetWrapper(RNNCell):
  '''Runs RNN inputs through a prenet before sending them to the cell.'''
  def __init__(self, cell, is_training, voice_print_feature=None, prenet_layer1=256, prenet_layer2=128):
    super(DecoderPrenetWrapper, self).__init__()
    self._cell = cell
    self._is_training = is_training
    self._voice_print_feature = voice_print_feature
    self._prenet_layer1 = prenet_layer1
    self._prenet_layer2 = prenet_layer2

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def call(self, inputs, state):
    if self._voice_print_feature is None:
      prenet_out = prenet(inputs, self._is_training, scope='decoder_prenet')
      return self._cell(prenet_out, state)
    else:
      print('inputs')
      print(inputs)
      print('fv')
      print(self._voice_print_feature)
      temp = tf.concat([inputs, self._voice_print_feature], 1)
      print(temp.get_shape())
      prenet_out = prenet(temp, self._is_training,
                          scope='decoder_prenet', layer_sizes=[self._prenet_layer1, self._prenet_layer2])

      #prenet_out2 = prenet(self._voice_print_feature, self._is_training, scope='decoder_prenet2')
      return self._cell(prenet_out, state)


  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)



class ConcatOutputAndAttentionWrapper(RNNCell):
  '''Concatenates RNN cell output with the attention context vector.

  This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
  attention_layer_size=None and output_attention=False. Such a cell's state will include an
  "attention" field that is the context vector.
  '''
  def __init__(self, cell):
    super(ConcatOutputAndAttentionWrapper, self).__init__()
    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size + self._cell.state_size.attention

  def call(self, inputs, state):
    output, res_state = self._cell(inputs, state)
    return tf.concat([output, res_state.attention], axis=-1), res_state

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)
