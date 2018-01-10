import numpy as np
import os
import random
import tensorflow as tf
import threading
import time
import traceback
from text import cmudict, text_to_sequence
from util.infolog import log


_batches_per_group = 32
_p_cmudict = 0.5
_pad = 0


class DataFeeder(threading.Thread):
  '''Feeds batches of data into a queue on a background thread.'''

  def __init__(self, coordinator, training_path, hparams):
    super(DataFeeder, self).__init__()
    self._coord = coordinator
    self._hparams = hparams
    self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    self._offset = 0
    self._offset_person_id = 0
    self._batch_in_queue = 0
    self._datasets = hparams.datasets

    # Load metadata:
    #self._datadir = os.path.dirname(metadata_filename)
    #with open(metadata_filename, encoding='utf-8') as f:
    #  self._metadata = [line.strip().split('|') for line in f]
    #  hours = sum((int(x[2]) for x in self._metadata)) * hparams.frame_shift_ms / (3600 * 1000)
    #  log('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))
    # self._metadata
    # self._datadir
    self._metadata = []
    global_parson_id = 0
    for dataset in self._datasets:
      metadata_filename = os.path.join(training_path, dataset, 'train.txt')
      datadir = os.path.dirname(metadata_filename)
      #exist_person_id correlate the global_person_id with current person_id
      exist_person_id = {}
      with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        hours = sum((int(x[2]) for x in metadata)) * hparams.frame_shift_ms / (3600 * 1000)
        log('Loaded ' + dataset + ' metadata for %d examples (%.2f hours)' % (len(metadata), hours))
        for item in metadata:
          #item=[vctk-spec-23918.npy,vctk-mel-23918.npy,329,They say that vital evidence was not heard in court.,60]
          person_id = item[4]
          item[0] = os.path.join(datadir, item[0])
          item[1] = os.path.join(datadir, item[1])
          if not person_id in exist_person_id:
            exist_person_id[person_id] = global_parson_id
            global_parson_id += 1
            self._metadata.append([])
          global_person_id_crrt = exist_person_id[person_id]
          self._metadata[global_person_id_crrt].append(item)







    # Create placeholders for inputs and targets. Don't specify batch size because we want to
    # be able to feed different sized batches at eval time.
    self._placeholders = [
      tf.placeholder(tf.int32, [None, None], 'inputs'),
      tf.placeholder(tf.int32, [None], 'input_lengths'),
      tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
      tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets')
    ]

    # Create queue for buffering data:
    queue = tf.FIFOQueue(100, [tf.int32, tf.int32, tf.float32, tf.float32], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    self.inputs, self.input_lengths, self.mel_targets, self.linear_targets = queue.dequeue()
    self.inputs.set_shape(self._placeholders[0].shape)
    self.input_lengths.set_shape(self._placeholders[1].shape)
    self.mel_targets.set_shape(self._placeholders[2].shape)
    self.linear_targets.set_shape(self._placeholders[3].shape)

    # Load CMUDict: If enabled, this will randomly substitute some words in the training data with
    # their ARPABet equivalents, which will allow you to also pass ARPABet to the model for
    # synthesis (useful for proper nouns, etc.)
    if hparams.use_cmudict:
      cmudict_path = os.path.join(self._datadir, 'cmudict-0.7b')
      if not os.path.isfile(cmudict_path):
        raise Exception('If use_cmudict=True, you must download ' +
          'http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b to %s'  % cmudict_path)
      self._cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=False)
      log('Loaded CMUDict with %d unambiguous entries' % len(self._cmudict))
    else:
      self._cmudict = None


  def start_in_session(self, session):
    self._session = session
    self.start()


  def run(self):
    try:
      while not self._coord.should_stop():
        self._enqueue_next_group()
    except Exception as e:
      traceback.print_exc()
      self._coord.request_stop(e)


  def _enqueue_next_group(self):
    start = time.time()

    # Read a group of examples:
    n = self._hparams.batch_size * self._hparams.num_GPU
    r = self._hparams.outputs_per_step

    print('current person id %d' % self._offset_person_id)
    # if the number of rest items is large
    if len(self._metadata[self._offset_person_id]) - self._offset > n * _batches_per_group:
      examples = [self._get_next_example() for i in range(n * _batches_per_group)]
    # if the of the rest is little, then read in the rest to generate moderate number of batches and update the
    # offset_person_id
    if len(self._metadata[self._offset_person_id]) - self._offset <= n * _batches_per_group:
      batches_per_group_crrt = int( (len(self._metadata[self._offset_person_id]) - self._offset)/n ) + 1
      examples = [self._get_next_example() for i in range(n * batches_per_group_crrt)]
      self._offset_person_id += 1
      self._offset = 0
      if self._offset_person_id >= len(self._metadata):
        self._offset_person_id = 0
        random.shuffle(self._metadata)




    # Bucket examples based on similar output sequence length for efficiency:
    examples.sort(key=lambda x: x[-1])
    batches = [examples[i:i+n] for i in range(0, len(examples), n)]
    #print('batches')
    #split the batches in case the big wav file induct Out Of Memory
    def split_list(list, n):
      output = []
      for i in range(n):
        output.append(list[int(len(list)*i/n): int(len(list)*(i+1)/n)])
      return output
    batches2 = []
    for i, batch in enumerate(batches):
      print('len of spec %d' % batches[i][-1][-1])
      if batches[i][-1][-1] / (self._hparams.max_iters * self._hparams.outputs_per_step) > 4:
        del(batches[i])
        #print(len(batches2))
      elif batches[i][-1][-1] / (self._hparams.max_iters * self._hparams.outputs_per_step) >2:
        del (batches[i])
        split_batch = split_list(batch, 4)
        for item in split_batch:
          batches2.append(item)
        #print(len(batches2))
      elif batches[i][-1][-1] / (self._hparams.max_iters * self._hparams.outputs_per_step) >1:
        del (batches[i])
        split_batch = split_list(batch, 2)
        for item in split_batch:
          batches2.append(item)
        #print(len(batches2))
      else:
        batches2.append(batch)
        #print(len(batches2))
    batches = batches2


    random.shuffle(batches)

    log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
    for batch in batches:
      feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
      self._session.run(self._enqueue_op, feed_dict=feed_dict)
      self._batch_in_queue += 1
      log('self._batch_in_queue: %s' % str(self._batch_in_queue), slack=True)




  def _get_next_example(self):
    '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
    linear_target = range(self._hparams.max_iters * self._hparams.outputs_per_step * 4 + 1)
    while len(linear_target) > self._hparams.max_iters * self._hparams.outputs_per_step * 4 - 1:
      if self._offset >= len(self._metadata[self._offset_person_id]):
        self._offset = 0
        random.shuffle(self._metadata[self._offset_person_id])

      meta = self._metadata[self._offset_person_id][self._offset]
      self._offset += 1

      text = meta[3]
      if self._cmudict and random.random() < _p_cmudict:
        text = ' '.join([self._maybe_get_arpabet(word) for word in text.split(' ')])

      input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
      linear_target = np.load(meta[0])
      mel_target = np.load(meta[1])

    return (input_data, mel_target, linear_target, len(linear_target))


  def _maybe_get_arpabet(self, word):
    arpabet = self._cmudict.lookup(word)
    return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word


def _prepare_batch(batch, outputs_per_step):
  random.shuffle(batch)
  inputs = _prepare_inputs([x[0] for x in batch])
  input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
  mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
  linear_targets = _prepare_targets([x[2] for x in batch], outputs_per_step)
  return (inputs, input_lengths, mel_targets, linear_targets)


def _prepare_inputs(inputs):
  max_len = max((len(x) for x in inputs))
  return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
  max_len = max((len(t) for t in targets)) + 1
  return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
  return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
  return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
  remainder = x % multiple
  return x if remainder == 0 else x + multiple - remainder
