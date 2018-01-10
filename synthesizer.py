import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio
from weight_norm_fbank_10782_20171124_low25_high3700_global_norm.ResCNN_wn_A_softmax_prelu_2deeper_group import ResCNN


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    mel_spec =  tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets')

    with tf.variable_scope('model') as scope:

      self.net = ResCNN(data=mel_spec, hyparam=hparams)
      self.net.inference()
      voice_print_feature = tf.reduce_mean(self.net.features, 0)

      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs=inputs, input_lengths=input_lengths, voice_print_feature=voice_print_feature)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text, mel_spec):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
      self.net.data2: mel_spec
    }
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    #wav = wav[:audio.find_endpoint(wav)]
    #out = io.BytesIO()
    #audio.save_wav(wav, out)
    return wav
