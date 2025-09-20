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

from pyannote.metrics.segmentation import Annotation, Segment

from pyannote_tsvad.pipelines.tsvad_diarization_rvector import SpeakerDiarization as TSVADSpeakerDiarizationPipeline
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

class PrepareSingleSpeakerWav:
    def __init__(
        self,
        min_segment_dur: float = 0.3,
        min_silence_dur: float = 0.02,
        max_silence_dur: float = 0.08,
        shuffle_segments: bool = True
    ):
        self.audio = Audio(sample_rate=16000, mono="downmix")

        self.min_segment_dur = min_segment_dur
        self.min_silence_dur = min_silence_dur
        self.max_silence_dur = max_silence_dur
        self.shuffle_segments = shuffle_segments

    def extract_clean_single_speaker_timeline(self, annotation, spk):
        spk_tl = annotation.label_timeline(spk)

        others_tl = Timeline()
        for other_spk in annotation.labels():
            if other_spk != spk:
                others_tl = others_tl | annotation.label_timeline(other_spk)

        overlap_with_others = spk_tl.crop(others_tl, mode="intersection")
        clean_tl = spk_tl.extrude(overlap_with_others)

        return clean_tl

    def process_single_file(self, wav_path, rttm_path):
        rec_id = wav_path.stem

        ann_by_file = load_rttm(rttm_path)
        if rec_id not in ann_by_file:
            raise ValueError(f"Missing annotation for {rec_id}")

        ann = ann_by_file[rec_id]
        all_tl: Timeline = ann.get_timeline().support()

        single_speaker_wav_dict = dict()

        for spk in ann.labels():
            clean_tl = self.extract_clean_single_speaker_timeline(ann, spk)

            filtered_segments = [segment for segment in clean_tl if segment.duration > self.min_segment_dur]
            if not filtered_segments:
                print(f"No segments longer than {self.min_segment_dur}s for {rec_id} {spk}")
                continue

            if self.shuffle_segments:
                random.shuffle(filtered_segments)

            wav_chunks = list()
            for i, segment in enumerate(filtered_segments):

                audio_duration = self.audio.get_duration(wav_path)
                if segment.start >= audio_duration:
                    print(f"Segment [{segment.start:.2f}s, {segment.end:.2f}s] starts after audio file ends ({audio_duration:.2f}s) for {wav_path}. Skipping...")
                    continue

                if segment.end > audio_duration:
                    original_end = segment.end
                    segment = Segment(start=segment.start, end=audio_duration)
                    print(f"Segment end time clipped from {original_end:.2f}s to {audio_duration:.2f}s for {wav_path}")

                if segment.end - segment.start < self.min_segment_dur:
                    print(f"Segment [{segment.start:.2f}s, {segment.end:.2f}s] too short after clipping for {wav_path}. Skipping...")
                    continue

                chunk, _ = self.audio.crop(wav_path, segment, mode="raise")
                wav_chunks.append(chunk.squeeze(0))

                if i < len(filtered_segments) - 1:
                    silence_duration = random.uniform(self.min_silence_dur, self.max_silence_dur)
                    gap = np.zeros(int(silence_duration * self.audio.sample_rate), dtype=np.float32)
                    wav_chunks.append(gap)

            spk_wav = np.concatenate(wav_chunks)
            single_speaker_wav_dict[spk] = spk_wav

        return single_speaker_wav_dict

class ExtractSingleSpeakerEmbedding:
    def __init__(
        self,
        model_path,
        duration=5.0,
        step=2.5,
        batch_size=32,
        device=torch.device("cpu"),
    ):
        self.duration = duration
        self.step = step
        self.batch_size = batch_size
        self.device = device

        self.infer = Inference(
            model=model_path,
            window="sliding",
            duration=duration,
            step=step,
            skip_aggregation=True,
            batch_size=batch_size,
            device=device,
        )
        # in case chunks are very short
        self.model = PretrainedSpeakerEmbedding(embedding=model_path, device=device)

    def crop_to_integer_chunks(self, waveform, sample_rate, duration, step):
        total_len = waveform.shape[1]
        chunk_size = int(duration * sample_rate)
        step_size = int(step * sample_rate)

        if total_len < chunk_size:
            raise ValueError

        num_chunks = 1 + (total_len - chunk_size) // step_size
        final_len = (num_chunks - 1) * step_size + chunk_size

        residual = total_len - final_len
        if residual > step_size // 2:
            return np.concatenate([
                waveform[:, :final_len],
                waveform[:, -chunk_size:]
            ], axis=1)
        else:
            return waveform[:, :final_len]

    def extract_from_wav(self, spk: str, waveform: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """Extract embeddings for a single waveform (in memory)"""
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

        if waveform.shape[-1] < self.duration * sample_rate:
            print(f"[{spk}] waveform too short: {waveform.shape[-1]/sample_rate:.2f}s")
            embeddings = self.model(torch.from_numpy(waveform).unsqueeze(0).to(self.device))
            return embeddings
        else:
            waveform = self.crop_to_integer_chunks(waveform, sample_rate, self.duration, self.step)
            embeddings = self.infer({"waveform": torch.from_numpy(waveform).to(self.device), "sample_rate": sample_rate})
            return embeddings

def diarize_session(
    sess_name,
    pipeline,  # diarization pipeline
    wav_files,
    uem_boundaries,
    init_rttm_file,  # initial RTTM file for this session
    single_speaker_preparation,  # Inited PrepareSingleSpeakerWav instance
    embedding_extractor,  # Inited ExtractSingleSpeakerEmbedding instance
    apply_median_filtering=False,
):
    print("Prepare single speaker wav for single speaker embedding extraction, using init_rttm_file...")
    wav_path, rttm_path = Path(wav_files[0]), Path(init_rttm_file)
    assert wav_path.exists() and rttm_path.exists()
    sess_single_speaker_wav_dict = single_speaker_preparation.process_single_file(
        wav_path=wav_path,
        rttm_path=rttm_path,
    )

    print(f"Extracting embedding on single speaker wavs...")
    sess_single_speaker_embedding_dict = dict()
    for spk, wav in sess_single_speaker_wav_dict.items():
        embeddings = embedding_extractor.extract_from_wav(spk, wav)
        sess_single_speaker_embedding_dict[spk] = embeddings

    print('Extracting segmentations and prepare speaker embeddings...')
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = torch.unsqueeze(waveform[0], 0)
    segmentations, sess_spk_perm, estimated_sess_spk = pipeline.get_segmentations(
        {"waveform": waveform, "sample_rate": sample_rate},
        estimated_speaker_embeddings=sess_single_speaker_embedding_dict,
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
        "--training_embedding_root",
        type=str,
        default="",
        help="Path to training embedding root.",
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
        training_embedding_root=args.training_embedding_root,
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

    prepare_single_speaker_wav = PrepareSingleSpeakerWav(
        min_segment_dur=0.3,
        min_silence_dur=0.02,
        max_silence_dur=0.08,
        shuffle_segments=True
    )
    embedding_extractor = ExtractSingleSpeakerEmbedding(
        model_path=args.embedding_model,
        duration=5.0,
        step=2.5,
        batch_size=args.batch_size,
        device=diarization_pipeline.device
    )

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
            single_speaker_preparation=prepare_single_speaker_wav,
            embedding_extractor=embedding_extractor,
            apply_median_filtering=args.apply_median_filtering
        )
        rttm_out = os.path.join(args.out_dir, sess + ".rttm")  # TS-VAD predicted rttm file after inference
        with open(rttm_out, "w") as f:
            f.write(diar_result.to_rttm())
