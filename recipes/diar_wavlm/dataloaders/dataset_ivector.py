# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
# Copyright 2025 Nanjing University (author: Zeyan Song, clovermaxszy@gmail.com)

import os
from pathlib import Path
import torch
import numpy as np

import soundfile as sf
from typing import Dict
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
import random
import copy

def load_scp(scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

def load_uem(uem_file: str) -> Dict[str, float]:
    """ returns dictionary { recid: duration }  """
    if not os.path.exists(uem_file):
        return None
    lines = [line.strip().split() for line in open(uem_file)]
    return {x[0]: [float(x[-2]), float(x[-1])] for x in lines}

def _gen_chunk_indices(
    init_posi: int,  # start_sec
    data_len: int,  # end_sec
    size: int,  # chunk_size
    step: int,  # chunk_shift
):
    init_posi = int(init_posi + 1)
    data_len = int(data_len - 1)
    cur_len = data_len - init_posi
    assert cur_len > size
    num_chunks = int((cur_len - size + step) / step)
    
    for i in range(num_chunks):
        yield init_posi + (i * step), init_posi + (i * step) + size

def get_dtype(value: int) -> str:
    """Return the most suitable type for storing the
    value passed in parameter in memory.

    Parameters
    ----------
    value: int
        value whose type is best suited to storage in memory

    Returns
    -------
    str:
        numpy formatted type
        (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
    """
    # signe byte (8 bits), signed short (16 bits), signed int (32 bits):
    types_list = [(127, "b"), (32_768, "i2"), (2_147_483_648, "i")]
    filtered_list = [
        (max_val, type) for max_val, type in types_list if max_val > abs(value)
    ]
    if not filtered_list:
        return "i8"  # signed long (64 bits)
    return filtered_list[0][1]

class LoadIVector():
    def __init__(self, speaker_embedding_txt):
        self.speaker_embedding = self.load_ivector(speaker_embedding_txt)

    def load_ivector(self, speaker_embedding_txt):
        SCP_IO = open(speaker_embedding_txt)
        speaker_embedding = {}
        raw_lines = [l for l in SCP_IO]
        SCP_IO.close()
        for i in tqdm(range(len(raw_lines) // 2), ncols=100):
            speaker = raw_lines[2*i].split()[0]
            ivector = np.array(raw_lines[2*i+1].split()[:-1], np.float32)
            speaker_embedding[speaker] = torch.from_numpy(ivector)
        return speaker_embedding

    def get_speaker_embedding(self, speaker):
        if not speaker in self.speaker_embedding.keys():
            # print("{} not in speaker embedding list".format(speaker))
            return None
        return self.speaker_embedding[speaker]

class TSVADDiarizationDataset(Dataset):
    def __init__(
        self, 
        scp_file: str, 
        rttm_file: str,
        uem_file: str,
        emb_txt: str,
        training_emb_txt: str,
        model_num_frames: int,
        model_rf_duration: float,
        model_rf_step: float,
        chunk_size: int,
        chunk_shift: int,
        max_speakers: int,
        sample_rate: int = 16000,
    ): 
        self.chunk_indices = []

        self.sample_rate = sample_rate
        self.max_speakers = max_speakers

        self.model_rf_step = model_rf_step
        self.model_rf_duration = model_rf_duration
        self.model_num_frames = model_num_frames

        self.rec_scp = load_scp(scp_file)
        self.reco2dur = load_uem(uem_file)
        self.unmatch_cnt = 0

        self.spk_embedding = LoadIVector(emb_txt)
        self.training_spk_embedding = LoadIVector(training_emb_txt)
        self.data_set_speaker = list(self.training_spk_embedding.speaker_embedding.keys())

        for idx, (rec, dur_info) in enumerate(self.reco2dur.items()):
            start_sec, end_sec = dur_info   
            try:
                if chunk_size > 0:
                    for st, ed in _gen_chunk_indices(
                            start_sec,
                            end_sec,
                            chunk_size,
                            chunk_shift
                    ):
                        self.chunk_indices.append((rec, self.rec_scp[rec], st, ed))
                else:
                    self.chunk_indices.append((rec, self.rec_scp[rec], start_sec, end_sec))
            except:
                print(f'Too short recording: {rec}, duration: {end_sec - start_sec} seconds')
                self.unmatch_cnt += 1

        print(f'Total unmatched recordings: {self.unmatch_cnt}, proportion: {(self.unmatch_cnt * 100 / len(self.reco2dur)):.4f}%')
        print(f'Total chunks: {len(self.chunk_indices)}')

        self.session_spk_map = defaultdict(dict)
        self.annotations = self.rttm2label(rttm_file, self.session_spk_map)

    def get_session_idx(self, session):
        """
        convert session to session idex
        """
        session_keys = list(self.rec_scp.keys())
        return session_keys.index(session)

    def rttm2label(self, rttm_file, session_spk_map):
        '''
        SPEAKER train100_306 1 15.71 1.76 <NA> <NA> 5456 <NA> <NA>
        '''
        annotations = []
        session_idx_cache = {}  # {session_name: session_idx}

        with open(rttm_file, 'r') as file:
            for line in tqdm(file, ncols=100):
                parts = line.strip().split()
                if len(parts) < 9:
                    continue  # skip invalid lines

                session = parts[1]
                start = float(parts[3])
                end = start + float(parts[4])
                spk = parts[-2] if parts[-2] != "<NA>" else parts[-3]

                # cache session idx
                if session not in session_idx_cache:
                    session_idx_cache[session] = self.get_session_idx(session)

                # cache speaker idx per session
                if spk not in session_spk_map[session]:
                    session_spk_map[session][spk] = len(session_spk_map[session])

                label_idx = session_spk_map[session][spk]
                annotations.append((session_idx_cache[session], start, end, label_idx))

        segment_dtype = [
            ("session_idx", get_dtype(max(a[0] for a in annotations))),
            ("start", "f"),
            ("end", "f"),
            ("label_idx", get_dtype(max(a[3] for a in annotations))),
        ]

        return np.array(annotations, dtype=segment_dtype)

    def extract_wavforms(self, path, start, end, num_channels=1):
        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)
        data, sample_rate = sf.read(path, start=start, stop=end, dtype='float32')
        assert sample_rate == self.sample_rate
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = np.einsum('tc->ct', data) 
        return data[:num_channels, :]

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        session, path, chunk_start, chunk_end = self.chunk_indices[idx]
        data = self.extract_wavforms(path, chunk_start, chunk_end)  # [start, end)

        all_spk_dict = self.session_spk_map[session]
        n_session_spk = len(all_spk_dict)  # number of speakers in this session

        # chunked annotations
        session_idx = self.get_session_idx(session)
        annotations_session = self.annotations[self.annotations['session_idx'] == session_idx]
        chunked_annotations = annotations_session[
            (annotations_session["start"] < chunk_end) & (annotations_session["end"] > chunk_start)
        ]

        mask_label = np.zeros((self.model_num_frames, n_session_spk), dtype=np.int64)

        # discretize chunk annotations at model output resolution (here frame resolution)
        step = self.model_rf_step
        half = 0.5 * self.model_rf_duration

        start = np.maximum(chunked_annotations["start"], chunk_start) - chunk_start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunked_annotations["end"], chunk_end) - chunk_start - half
        end_idx = np.round(end / step).astype(int)

        for start, end, label in zip(
            start_idx, end_idx, chunked_annotations['label_idx']  # speaking region
        ):
            mask_label[start : end + 1, int(label)] = 1

        sorted_all_spk_dict = sorted(all_spk_dict, key=lambda k: all_spk_dict[k])
        emb_list = list()
        for spk in sorted_all_spk_dict:
            embeddings = self.spk_embedding.get_speaker_embedding(f"{session}-{spk}")
            if embeddings is not None:
                emb_list.append(embeddings)
            else:
                exist_keys = {f"{session}-{spk}" for spk in all_spk_dict.keys()}
                cand = [k for k in self.data_set_speaker if k not in exist_keys]
                rand_keys = random.sample(cand, 1)
                append_spk = self.training_spk_embedding.get_speaker_embedding(rand_keys[0])
                emb_list.append(append_spk)
        emb_data = np.stack(emb_list, axis=0)  # [n_session_spk, emb_dim]

        if n_session_spk > self.max_speakers:
            spk_dur = np.sum(mask_label, axis=0)
            keep_idx = np.argsort(-spk_dur)[: self.max_speakers]  # talktive order
            mask_label = mask_label[:, keep_idx]
            emb_data = emb_data[keep_idx]
        elif n_session_spk < self.max_speakers:
            pad_speakers = self.max_speakers - n_session_spk
            mask_label = np.pad(mask_label, ((0, 0),(0, pad_speakers)), mode='constant')

            exist_keys = {f"{session}-{spk}" for spk in all_spk_dict.keys()}
            cand = [k for k in self.data_set_speaker if k not in exist_keys]
            rand_keys = random.sample(cand, pad_speakers)

            append_spk_list = [self.training_spk_embedding.get_speaker_embedding(k) for k in rand_keys]
            emb_data = np.concatenate([emb_data, np.stack(append_spk_list)], axis=0)
        else:
            pass

        return data, emb_data, mask_label, session

    def collater(self, batch):
        collated_x = []
        collated_x_emb = []
        collated_y = []
        collated_names = []

        for x, x_emb, y, name in batch:
            perm = np.random.permutation(self.max_speakers)
            x_emb = x_emb[perm]
            y = y[:, perm]

            collated_x.append(x)
            collated_x_emb.append(x_emb)
            collated_y.append(y)
            collated_names.append(name)

        return {
            "xs": torch.from_numpy(np.stack(collated_x)).float(),  # (batch, 1, num_samples)
            "xs_emb": torch.from_numpy(np.stack(collated_x_emb)).float(),  # (batch, max_speakers, embedding_dim)
            "ts": torch.from_numpy(np.stack(collated_y)),  # (batch, num_frames, max_speakers)
            "names": collated_names
        }


if __name__ == '__main__':

    from tqdm import tqdm
    dataset = TSVADDiarizationDataset(
        scp_file = "/data/ssd1/zeyan.song/DATA_Compound/data_sc/eval/wav.scp",
        rttm_file = "/data/ssd1/zeyan.song/DATA_Compound/data_sc/eval/rttm",
        uem_file = "/data/ssd1/zeyan.song/DATA_Compound/data_sc/eval/all.uem",
        emb_txt = "/data/ssd1/zeyan.song/DATA_Compound/exp_data/NSD-MS2S/metadata/eval/ivector_oracle/ivectors_spk.txt",
        training_emb_txt = "/data/ssd1/zeyan.song/DATA_Compound/exp_data/NSD-MS2S/metadata/train/ivector_oracle/ivectors_spk.txt",
        model_num_frames=799,
        model_rf_duration=0.025,
        model_rf_step=0.02,
        chunk_size=8,
        chunk_shift=6,
        max_speakers=10,
        sample_rate=16000,
    )

    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=8,
        num_workers=0,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=dataset.collater
    )

    for step, batch in enumerate(tqdm(dataloader, ncols=100)):
        pass
        # if step == 5:
        #     break