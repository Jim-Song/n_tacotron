from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from hparams import hparams
from util import audio
import re


_max_out_length = 700
_end_buffer = 0.05
_min_confidence = 90


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the vctk dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1

  VCTK_path = in_dir
  VCTK_path_wav = os.path.join(VCTK_path, 'wav48')
  VCTK_path_txt = os.path.join(VCTK_path, 'txt')

  person_id = 0

  for dir_name in os.listdir(VCTK_path_wav):
    # dir_name = 'p300'
    # wav_dir  = '/home/pattern/songjinming/tts/data/VCTK-Corpus/wav48/p300'
    # txt_dir  = ''/home/pattern/songjinming/tts/data/VCTK-Corpus/txt/p300''
    wav_dir = os.path.join(VCTK_path_wav, dir_name)
    txt_dir = os.path.join(VCTK_path_txt, dir_name)

    person_id += 1

    for wav_file in os.listdir(wav_dir):
      # wav_file  = 'p300_224.wav'
      # name_file = 'p300_224'
      # txt_file  = 'p300_224.txt'
      # wav_root  = '/home/pattern/songjinming/tts/data/VCTK-Corpus/wav48/p300/p300_224.wav'
      # txt_root  = '/home/pattern/songjinming/tts/data/VCTK-Corpus/txt/p300/p300_224.txt'
      name_file = os.path.splitext(wav_file)[0]
      # some file is not wav file and just skip
      # print(os.path.splitext(wav_file))
      if not os.path.splitext(wav_file)[1] == '.wav':
        continue
      txt_file = '.'.join([name_file, 'txt'])
      wav_root = os.path.join(wav_dir, wav_file)
      txt_root = os.path.join(txt_dir, txt_file)
      # audio
      # print(wav_root)
      # f_wav = wave.open(wav_root, 'rb')
      # params = f_wav.getparams()
      # sample_rate = params[2]
      # n_samples = params[3]
      # txt
      # some wav files dont have correspond txt file
      try:
        text = open(txt_root, 'r').read()
        text = re.sub('\n', '', text)
      except:
        continue

      futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_root, text, person_id)))

      index += 1


  return [future.result() for future in tqdm(futures)]

def _process_utterance(out_dir, index, wav_path, text, person_id):
  '''Preprocesses a single utterance audio/text pair.
  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.
  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file
  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''
  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)
  # Compte the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]
  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
  # Write the spectrograms to disk:
  spectrogram_filename = 'vctk-spec-%05d.npy' % index
  mel_filename = 'vctk-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text, person_id)








