from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from hparams import hparams
from util import audio
import re


_max_out_length = 700
_end_buffer = 0.05
_min_confidence = 50

# Note: "A Tramp Abroad" & "The Man That Corrupted Hadleyburg" are higher quality than the others.
books = [
    'ATrampAbroad',
    'TheManThatCorruptedHadleyburg',
    'LifeOnTheMississippi',
    'TheAdventuresOfTomSawyer',
]

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    arctic_dirs = [os.path.join(in_dir, 'cmu_us_awb_arctic-0.90-release/cmu_us_awb_arctic'),
                    os.path.join(in_dir, 'cmu_us_bdl_arctic-0.95-release/cmu_us_bdl_arctic'),
                    os.path.join(in_dir, 'cmu_us_clb_arctic-0.95-release/cmu_us_clb_arctic'),
                    os.path.join(in_dir, 'cmu_us_jmk_arctic-0.95-release/cmu_us_jmk_arctic'),
                    os.path.join(in_dir, 'cmu_us_ksp_arctic-0.95-release/cmu_us_ksp_arctic'),
                    os.path.join(in_dir, 'cmu_us_rms_arctic-0.95-release/cmu_us_rms_arctic'),
                    os.path.join(in_dir, 'cmu_us_slt_arctic-0.95-release/cmu_us_slt_arctic')
                    ]
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    person_id = 0
    for dir in arctic_dirs:
        person_id += 1
        with open(os.path.join(dir, 'etc', 'txt.done.data')) as f:
            wav_dir = os.path.join(dir, 'wav')
            for line in f:
                name_file = re.findall('arctic_[ab][0-9]+', line)[0]  # 'arctic_a0001'
                text = re.findall('"(.*)"', line)[0]  # 'AUTHOR OF THE DANGER TRAIL, PHILIP STEELS, ETC'
                # get sample rate
                wav_path = os.path.join(dir, 'wav', '%s.wav' % name_file)
                # wav_path = '/home/pattern/songjinming/tts/data/arctic/cmu_us_awb_arctic-0.90-release/
                #           cmu_us_awb_arctic/wav/arctic_a0001.wav'
                task = partial(_process_utterance, out_dir, index, wav_path, text, person_id)
                futures.append(executor.submit(task))
                index += 1
    results = [future.result() for future in tqdm(futures)]
    return [r for r in results if r is not None]


def _process_utterance(out_dir, index, wav_path, text, person_id):
    # Load the wav file and trim silence from the ends:
    wav = audio.load_wav(wav_path)
    #max_samples = _max_out_length * hparams.frame_shift_ms / 1000 * hparams.sample_rate
    #if len(wav) > max_samples:
    #    return None
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    spectrogram_filename = 'arctic-spec-%05d.npy' % index
    mel_filename = 'arctic-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    return (spectrogram_filename, mel_filename, n_frames, text, person_id)


def _parse_labels(path):
    labels = []
    with open(os.path.join(path)) as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 3:
                labels.append((float(parts[0]), ' '.join(parts[2:])))
    start = 0
    end = None
    if labels[0][1] == 'sil' or labels[0][2] == 'sil':
        start = labels[0][0]
    if labels[-1][1] == 'sil' or labels[-1][2] == 'sil':
        end = labels[-2][0] + _end_buffer
    return (start, end)
