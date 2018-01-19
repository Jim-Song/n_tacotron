from util import audio
import argparse
from synthesizer import Synthesizer
import os
import numpy as np
from hparams import hparams






sentences = [
  # From July 8, 2017 New York Times:
  'Scientists at the CERN laboratory say they have discovered a new particle.',
  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
  'President Trump met with other leaders at the Group of 20 conference.',
  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
  'Generative adversarial network or variational auto-encoder.',
  'The buses aren\'t the problem, they actually provide a solution.',
  'Does the quick brown fox jump over the lazy dog?',
  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]



def main():
  parser = argparse.ArgumentParser()

  #parser.add_argument('--base_dir', default=os.path.expanduser('./'))
  parser.add_argument('--wav_path', default='./wav_files', help='the wav files to be minic')
  parser.add_argument('--output_dir', default='./synthesis', help='the output dir')
  parser.add_argument('--output_prefix', default=' ', help='the prefix of the name of the output')
  parser.add_argument('--model_path', default=' ', help='path of the trained model')
  parser.add_argument('--prenet_layer1', default=256, type=int, help='batch_size')  #
  parser.add_argument('--prenet_layer2', default=128, type=int, help='batch_size')  #
  parser.add_argument('--gru_size', default=256, type=int, help='batch_size')  #
  parser.add_argument('--attention_size', default=256, type=int, help='batch_size')  #
  parser.add_argument('--rnn_size', default=256, type=int, help='batch_size')  #

  args = parser.parse_args()

  hparams.prenet_layer1 = args.prenet_layer1
  hparams.prenet_layer2 = args.prenet_layer2
  hparams.gru_size = args.gru_size
  hparams.attention_size = args.attention_size
  hparams.rnn_size = args.rnn_size

  #log_dir = os.path.join(args.base_dir, 'logs-%s-%s' % (run_name, args.description))
  os.makedirs(os.path.join(args.output_dir, args.output_prefix), exist_ok=True)

  mel_spectrograms = []
  for wav_file in os.listdir(args.wav_path):

    # Load the audio to a numpy array:
    wav = audio.load_wav(os.path.join(args.wav_path, wav_file))

    # Compute the linear-scale spectrogram from the wav:
    # spectrogram = audio.spectrogram(wav).astype(np.float32)

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    mel_spectrograms.append(mel_spectrogram)
    print(wav_file)
    print(np.shape(mel_spectrogram))

  print(np.shape(mel_spectrograms))

  mel_spectrograms = _prepare_targets(mel_spectrograms, 1)

  synthesizer = Synthesizer(hparams)
  synthesizer.load(args.model_path)

  for text in sentences:

    wav = synthesizer.synthesize(text=text, mel_spec=mel_spectrograms)

    out = os.path.join(args.output_dir, args.output_prefix, text+'.wav')
    audio.save_wav(wav, out)

_pad = 0

def _prepare_targets(targets, alignment):
  max_len = max((len(t) for t in targets)) + 1
  return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])

def _pad_target(t, length):
  return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)

def _round_up(x, multiple):
  remainder = x % multiple
  return x if remainder == 0 else x + multiple - remainder































if __name__ == '__main__':
  main()
