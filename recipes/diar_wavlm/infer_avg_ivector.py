# Copyright 2025 Nanjing University (author: Zeyan Song, clovermaxszy@gmail.com)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import argparse
from omegaconf import OmegaConf
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.ndimage import median_filter
from tqdm import tqdm

from pyannote.metrics.segmentation import Annotation, Segment

from pyannote_tsvad.pipelines.tsvad_diarization_ivector import SpeakerDiarization as TSVADSpeakerDiarizationPipeline
from pyannote.audio.utils.signal import Binarize
from pyannote.audio import Inference
from pyannote.audio import Audio
from pyannote.audio.utils.signal import binarize
from pyannote.core import Timeline, SlidingWindowFeature
from pyannote.database.util import load_rttm
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

import random

def scp2path(scp_file):
    """ return path list """
    lines = [line.strip().split()[1] for line in open(scp_file)]
    return lines

def load_rttms(rttm_dir):
    """ return session to RTTM file mapping """
    sess2rttm = {}
    for rttm_file in Path(rttm_dir).glob("*.rttm"):
        sess_name = rttm_file.stem
        sess2rttm[sess_name] = str(rttm_file)
    return sess2rttm

def read_uem(uem_file):
    with open(uem_file, "r") as f:
        lines = f.readlines()
    lines = [x.rstrip("\n") for x in lines]
    uem2sess = {}
    for x in lines:
        sess_id, _, start, stop = x.split(" ")
        uem2sess[sess_id] = (float(start), float(stop))
    return uem2sess

def load_metric_summary(metric_file, ckpt_path):
    with open(metric_file, "r") as f:
        lines = f.readlines()
    out_lst = []
    for line in lines:
        assert "Validation Loss/DER" in line
        if "Validation Loss/DER/FA/Miss/Confusion/Miss_FA_sum" in line:
            epoch = line.split()[4].split(':')[0]
            Loss, DER = line.split()[5], line.split()[7]
            bin_path = f"epoch_{str(epoch).zfill(4)}/pytorch_model.bin"
            out_lst.append({
                'epoch': int(epoch),
                'bin_path': ckpt_path / bin_path,
                'Loss': float(Loss),
                'DER': float(DER)
            })
        else:
            print('Not standard metric summary file, please check!')
            epoch = line.split()[4].split(':')[0]
            Loss, DER = line.split()[-3], line.split()[-1]
            bin_path = f"epoch_{str(epoch).zfill(4)}/pytorch_model.bin"
            out_lst.append({
                'epoch': int(epoch),
                'bin_path': ckpt_path / bin_path,
                'Loss': float(Loss),
                'DER': float(DER)
            })
    return out_lst

class LoadIVector:
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

    def get_speakers_by_session(self, session_name):
        session_speakers = {}
        
        for speaker, embedding in self.speaker_embedding.items():
            if speaker.startswith(f"{session_name}-"):
                spk = speaker.split("-")[1]
                session_speakers[spk] = embedding
        
        return session_speakers

def diarize_session(
    sess_name,
    pipeline,  # diarization pipeline
    wav_files,
    uem_boundaries,
    init_rttm_file,  # initial RTTM file for this session
    apply_median_filtering=False,
):
    wav_path, rttm_path = Path(wav_files[0]), Path(init_rttm_file)
    assert wav_path.exists() and rttm_path.exists()

    print('Extracting segmentations and prepare speaker embeddings...')
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = torch.unsqueeze(waveform[0], 0)
    segmentations, sess_spk_perm, estimated_sess_spk = pipeline.get_segmentations(
        {"waveform": waveform, "sample_rate": sample_rate},
        session_name=sess_name,
    )

    if apply_median_filtering:  # 11 points median filtering
        print('Apply median filtering to segmentations...')
        segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

    # binarize segmentation
    binarized_segmentations: SlidingWindowFeature = binarize(
        segmentations,
        onset=pipeline.segmentation.threshold,
        initial_state=False,
    )

    # estimate frame-level number of instantaneous speakers
    count = pipeline.speaker_count(
        binarized_segmentations,
        pipeline._segmentation.model._receptive_field,
        warm_up=(0.0, 0.0),
    )  # frame-wise count of speaker numbers, frame resolution SlidingWindow object

    # during counting, we could possibly overcount the number of instantaneous
    # speakers due to segmentation errors, so we cap the maximum instantaneous number
    # of speakers
    count.data = np.minimum(count.data, estimated_sess_spk).astype(np.int8)

    continuous_diarization = pipeline.reconstruct(
        segmentations,  # SlidingWindowFeature with chunk resolution SlidingWindow
        sess_spk_perm,  # map valid speaker indices from shuffled "valid + pesudo" speaker indices
        estimated_sess_spk,  # number of valid speaker embeddings
        count,  # SlidingWindowFeature with frame resolution SlidingWindow
    )

    # convert to annotation
    to_annotation = Binarize(
        onset=0.5,
        offset=0.5,
        min_duration_on=0.0,
        min_duration_off=pipeline.segmentation.min_duration_off
    )
    result = to_annotation(continuous_diarization)  # pyannote.core.annotation.Annotation object
    offset = uem_boundaries[0] / sample_rate
    new_annotation = Annotation(uri=sess_name)
    speakers = result.labels()
    for spk in speakers:
        for seg in result.label_timeline(spk):
            new_annotation[Segment(seg.start + offset, seg.end + offset)] = spk

    return new_annotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This script performs diarization using Pyannote audio diarization pipeline ",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required arguments
    parser.add_argument(
        "--configuration",
        type=str,
        required=True,
        help="Configuration (*.yaml).",
    )
    parser.add_argument(
        "--in_wav_scp",
        type=str,
        required=True,
        help="test wav.scp.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--uem_file",
        type=str,
        required=True,
        help="Path to uem file.",
    )
    parser.add_argument(
        "--init_rttms",
        required=True,
        type=str,
        help="Path to init rttm label dir.",
    )
    parser.add_argument(
        "--embedding_model",
        required=True,
        type=str,
        help="Path to pretrained speaker embedding model.",
    )

    # Optional arguments
    parser.add_argument(
        "--avg_ckpt_num",
        type=int,
        default=5,
        help="the number of chckpoints of model averaging",
    )
    parser.add_argument(
        "--val_metric",
        type=str,
        default="Loss",
        help="validation metric",
        choices=["Loss", "DER"],
    )
    parser.add_argument(
        "--val_mode",
        type=str,
        default="best",
        help="validation metric mode",
        choices=["best", "prev", "center"],
    )
    parser.add_argument(
        "--val_metric_summary",
        type=str,
        default="",
        help="val_metric_summary",
    )
    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="",
        help="Path to pretrained segmentation model.",
    )
    parser.add_argument(
        "--segmentation_duration",
        type=float,
        default=8.0,
        help="Duration of segmentation window in seconds.",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size used for segmentation and embeddings extraction in inference.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=8,
        help="Maximum number of speakers in the session.",
    )
    parser.add_argument(
        "--spk_embedding_txt",
        type=str,
        default="",
        help="Path to eval speaker embedding text file.",
    )
    parser.add_argument(
        "--training_spk_embedding_txt",
        type=str,
        default="",
        help="Path to training speaker embedding text file.",
    )
    parser.add_argument(
        "--apply_median_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply median filtering to segmentation output.",
    )
    parser.add_argument(
        "--binarize_threshold",
        type=float,
        default=0.5,
        help="Threshold for binarization of segmentation output.",
    )

    args = parser.parse_args()
    print(args)

    config_path = Path(args.configuration).expanduser().absolute()
    config = OmegaConf.load(config_path.as_posix())

    ckpt_path = config_path.parent / 'checkpoints'
    if args.val_metric_summary:
        val_metric_lst = load_metric_summary(args.val_metric_summary, ckpt_path)
        val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[args.val_metric])
        best_val_metric_idx = val_metric_lst.index(val_metric_lst_sorted[0])
        if args.val_mode == "best":
            # print(f'averaging the best {args.avg_ckpt_num} checkpoints...')
            segmentation = val_metric_lst_sorted[:args.avg_ckpt_num]
        elif args.val_mode == "prev":
            # print(f'averaging previous {args.avg_ckpt_num} checkpoints to the converged moment...')
            segmentation = val_metric_lst[
                best_val_metric_idx - args.avg_ckpt_num + 1 :
                best_val_metric_idx + 1
            ]
        else:
            # print(f'averaging {args.avg_ckpt_num} checkpoints around the converged moment...')
            segmentation = val_metric_lst[
                best_val_metric_idx - args.avg_ckpt_num // 2 :
                best_val_metric_idx + args.avg_ckpt_num // 2 + 1
            ]
        assert len(segmentation) == args.avg_ckpt_num
    else:
        segmentation = args.segmentation_model

    # create, instantiate and apply the pipeline
    diarization_pipeline = TSVADSpeakerDiarizationPipeline(
        config=config,  # model configurations
        segmentation=segmentation,  # pretrained segmentation model, here is chunk-wise TS-VAD
        segmentation_duration=args.segmentation_duration,  # default chunk size
        segmentation_step=args.segmentation_step,  # default 90% overlap between two consecuive windows
        segmentation_batch_size=args.batch_size,
        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        max_speakers=args.max_speakers,
        speaker_embedding_txt=args.spk_embedding_txt,
        training_speaker_embedding_txt=args.training_spk_embedding_txt,
    )

    PIPELINE_PARAMS = {
        "segmentation": {
            "min_duration_off": 0.0,
            "threshold": args.binarize_threshold,  # threshold for binarization
        },
    }

    diarization_pipeline.instantiate(PIPELINE_PARAMS)

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)

    audio_f = scp2path(args.in_wav_scp)

    assert args.uem_file is not None  # need audio file length
    uem_map = read_uem(args.uem_file)
    sess2audio = {}
    for audio_file in audio_f:
        sess_name = Path(audio_file).stem.split('.wav')[0]
        if sess_name not in sess2audio.keys():
            sess2audio[sess_name] = []
        sess2audio[sess_name].append(audio_file)

    sess2rttm = load_rttms(args.init_rttms)

    # now for each session
    for sess in sess2audio.keys():
        assert sess in sess2rttm.keys(), f"Session {sess} not found in RTTM files."
        print("Diarizing Session {}".format(sess))
        c_uem = uem_map[sess]
        diar_result = diarize_session(
            sess_name=sess,
            pipeline=diarization_pipeline,
            wav_files=sess2audio[sess],
            uem_boundaries=c_uem,
            init_rttm_file=sess2rttm[sess],  # init rttm file for this session
            apply_median_filtering=args.apply_median_filtering
        )
        rttm_out = os.path.join(args.out_dir, sess + ".rttm")  # TS-VAD predicted rttm file after inference
        with open(rttm_out, "w") as f:
            f.write(diar_result.to_rttm())
