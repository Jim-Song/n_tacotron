import io
import numpy as np
import tensorflow as tf
#from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio
from weight_norm_fbank_10782_20171124_low25_high3700_global_norm.ResCNN_wn_A_softmax_prelu_2deeper_group import ResCNN


class Synthesizer:

  def __init__(self, hparams):
    self.hparams = hparams


  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    mel_spec =  tf.placeholder(tf.float32, [None, None, self.hparams.num_mels], 'mel_targets')

    with tf.variable_scope('model') as scope:

      if self.hparams.enable_fv1 or self.hparams.enable_fv2:
        self.net = ResCNN(data=mel_spec, hyparam=self.hparams)
        self.net.inference()

        voice_print_feature = tf.reduce_mean(self.net.features, 0)
      else:
        voice_print_feature = None

      self.model = create_model(model_name, self.hparams)
      self.model.initialize(inputs=inputs, input_lengths=input_lengths, voice_print_feature=voice_print_feature)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text, mel_spec):
    cleaner_names = [x.strip() for x in self.hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    if self.hparams.enable_fv1 or self.hparams.enable_fv2:
      feed_dict = {
        self.model.inputs: [np.asarray(seq, dtype=np.int32)],
        self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
        self.net.data2: mel_spec
      }
    else:
      feed_dict = {self.model.inputs: [np.asarray(seq, dtype=np.int32)],
                   self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),}
    wav, alignment = self.session.run([self.wav_output, self.model.alignments], feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    alignment = alignment[0]
    #wav = wav[:audio.find_endpoint(wav)]
    #out = io.BytesIO()
    #audio.save_wav(wav, out)
    return wav, alignment
