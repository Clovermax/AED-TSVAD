# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright 2025 Nanjing University (author: Zeyan Song, clovermaxszy@gmail.com)

import warnings
from pathlib import Path
from typing import Callable, List, Optional, Text, Tuple, Union, Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pytorch_lightning.utilities.memory import is_oom_error
from tqdm import tqdm

from pyannote.audio.core.io import AudioFile
from pyannote_tsvad.core.model import Model
from pyannote_tsvad.core.task import Specifications, Resolution
from pyannote_tsvad.utils.multi_task import map_with_specifications
from pyannote.audio.utils.reproducibility import fix_reproducibility


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
                spk = speaker.replace(f"{session_name}-", "")
                session_speakers[spk] = embedding

        return session_speakers


class BaseInference:
    pass


class Inference(BaseInference):
    """Inference

    Parameters
    ----------
    model : Model
        Model. Will be automatically set to eval() mode and moved to `device` when provided.
    window : {"sliding", "whole"}, optional
        Use a "sliding" window and aggregate the corresponding outputs (default)
        or just one (potentially long) window covering the "whole" file or chunk.
    duration : float, optional
        Chunk duration, in seconds. Defaults to duration used for training the model.
        Has no effect when `window` is "whole".
    step : float, optional
        Step between consecutive chunks, in seconds. Defaults to warm-up duration when
        greater than 0s, otherwise 10% of duration. Has no effect when `window` is "whole".
    pre_aggregation_hook : callable, optional
        When a callable is provided, it is applied to the model output, just before aggregation.
        Takes a (num_chunks, num_frames, dimension) numpy array as input and returns a modified
        (num_chunks, num_frames, other_dimension) numpy array passed to overlap-add aggregation.
    skip_aggregation : bool, optional
        Do not aggregate outputs when using "sliding" window. Defaults to False.
    skip_conversion: bool, optional
        In case a task has been trained with `powerset` mode, output is automatically
        converted to `multi-label`, unless `skip_conversion` is set to True.
    batch_size : int, optional
        Batch size. Larger values (should) make inference faster. Defaults to 32.
    device : torch.device, optional
        Device used for inference. Defaults to `model.device`.
        In case `device` and `model.device` are different, model is sent to device.
    use_auth_token : str, optional
        When loading a private huggingface.co model, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`
    """

    def __init__(
        self,
        model: Union[Model, Text, Path],
        window: Text = "sliding",
        duration: Optional[float] = None,
        step: Optional[float] = None,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        skip_aggregation: bool = False,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        speaker_embedding_txt: Optional[Text] = None,
        training_speaker_embedding_txt: Optional[Text] = None,
        max_speakers: int = 10,
    ):
        # ~~~~ model ~~~~~

        assert isinstance(model, Model)
        self.model = model

        self.device = device

        self.model.eval()
        self.model.to(self.device)

        specifications = self.model.specifications

        # ~~~~ sliding window ~~~~~

        if window not in ["sliding", "whole"]:
            raise ValueError('`window` must be "sliding" or "whole".')

        if window == "whole" and any(
            s.resolution == Resolution.FRAME for s in specifications
        ):
            warnings.warn(
                'Using "whole" `window` inference with a frame-based model might lead to bad results '
                'and huge memory consumption: it is recommended to set `window` to "sliding".'
            )
        self.window = window

        training_duration = next(iter(specifications)).duration
        duration = duration or training_duration
        if training_duration != duration:
            warnings.warn(
                f"Model was trained with {training_duration:g}s chunks, and you requested "
                f"{duration:g}s chunks for inference: this might lead to suboptimal results."
            )
        self.duration = duration

        training_max_speakers = len(next(iter(specifications)).classes)
        max_speakers = max_speakers or training_max_speakers
        if training_max_speakers != max_speakers:
            warnings.warn(
                f"Model was trained with {training_max_speakers} speakers, and you requested "
                f"{max_speakers} speakers for inference: this might lead to suboptimal results."
            )
        self.max_speakers = max_speakers

        self.ivector_loader = LoadIVector(speaker_embedding_txt=speaker_embedding_txt)
        self.training_ivector_loader = LoadIVector(speaker_embedding_txt=training_speaker_embedding_txt)
        self.data_set_speaker = list(self.training_ivector_loader.speaker_embedding.keys())

        # identity mapping when without powerset training
        conversion = list()
        for s in specifications:
            c = nn.Identity()
            conversion.append(c.to(self.device))

        if isinstance(specifications, Specifications):
            self.conversion = conversion[0]
        else:
            self.conversion = nn.ModuleList(conversion)

        # ~~~~ overlap-add aggregation ~~~~~

        self.skip_aggregation = skip_aggregation  # Do not aggregate outputs when using "sliding" window
        self.pre_aggregation_hook = pre_aggregation_hook

        self.warm_up = next(iter(specifications)).warm_up
        # Use that many seconds on the left- and rightmost parts of each chunk
        # to warm up the model. While the model does process those left- and right-most
        # parts, only the remaining central part of each chunk is used for aggregating
        # scores during inference.

        # step between consecutive chunks
        step = step or (
            0.1 * self.duration if self.warm_up[0] == 0.0 else self.warm_up[0]
        )

        if step > self.duration:
            raise ValueError(
                f"Step between consecutive chunks is set to {step:g}s, while chunks are "
                f"only {self.duration:g}s long, leading to gaps between consecutive chunks. "
                f"Either decrease step or increase duration."
            )
        self.step = step

        self.batch_size = batch_size

    def to(self, device: torch.device) -> "Inference":
        """Send internal model to `device`"""

        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.model.to(device)
        self.conversion.to(device)
        self.device = device
        return self

    def infer(
        self,
        chunks: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Forward pass

        Takes care of sending chunks (in a batch) to right device and outputs back to CPU

        Parameters
        ----------
        chunks : (batch_size, num_channels, num_samples) torch.Tensor
            Batch of audio chunks.
        embeddings: (batch_size, max_speakers, embedding_dim) torch.Tensor
            Batch of speaker embeddings.
            Each chunk, pseudo embeddings are randomly chosen from the training set, 
            and stack in random order with real estimated embeddings.

        Returns
        -------
        outputs : (tuple of) (batch_size, ...) np.ndarray
            Model output.
        """

        with torch.inference_mode():
            try:
                outputs = self.model(chunks.to(self.device), embeddings.to(self.device))  # [B, T, N]
            except RuntimeError as exception:
                if is_oom_error(exception):
                    raise MemoryError(
                        f"batch_size ({self.batch_size: d}) is probably too large. "
                        f"Try with a smaller value until memory error disappears."
                    )
                else:
                    raise exception
                
        def __convert(output: torch.Tensor, conversion: nn.Module, **kwargs):
            return conversion(output).cpu().numpy()

        return map_with_specifications(
            self.model.specifications, __convert, outputs, self.conversion
        )  # convert powerset classes to multi-label classes

    def slide(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        estimated_speaker_embeddings: torch.Tensor,
        hook: Optional[Callable],
    ) -> Union[SlidingWindowFeature, Tuple[SlidingWindowFeature]]:
        """Slide model on a waveform

        Parameters
        ----------
        waveform: (num_channels, num_samples) torch.Tensor
            Waveform.
        sample_rate : int
            Sample rate.
        estimated_speaker_embeddings : torch.Tensor
            Estimated speaker embeddings stacked and shuffled with pseudo embeddings chosen from training set.
            shape: (max_speakers, embedding_dim)
        hook: Optional[Callable]
            When a callable is provided, it is called everytime a batch is
            processed with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature
            Model output. Shape is (num_chunks, dimension) for chunk-level tasks,
            and (num_frames, dimension) for frame-level tasks.
        """

        window_size: int = self.model.audio.get_num_samples(self.duration)
        step_size: int = round(self.step * sample_rate)
        _, num_samples = waveform.shape

        def __frames(
            receptive_field, specifications: Optional[Specifications] = None
        ) -> SlidingWindow:
            if specifications.resolution == Resolution.CHUNK:
                return SlidingWindow(start=0.0, duration=self.duration, step=self.step)
            return receptive_field  # SlidingWindow object

        frames: Union[SlidingWindow, Tuple[SlidingWindow]] = map_with_specifications(
            self.model.specifications, __frames, self.model._receptive_field
        )

        # prepare complete chunks
        if num_samples >= window_size:
            chunks: torch.Tensor = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk_num frame_length -> chunk_num channel frame_length",
            )  # GET CHUNK HERE
            num_chunks, _, _ = chunks.shape
        else:
            num_chunks = 0

        # prepare last incomplete chunk
        has_last_chunk = (num_samples < window_size) or (
            num_samples - window_size
        ) % step_size > 0
        if has_last_chunk:
            # pad last chunk with zeros
            last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]
            _, last_window_size = last_chunk.shape
            last_pad = window_size - last_window_size
            last_chunk = F.pad(last_chunk, (0, last_pad))

        def __empty_list(**kwargs):
            return list()

        outputs: Union[
            List[np.ndarray], Tuple[List[np.ndarray]]
        ] = map_with_specifications(self.model.specifications, __empty_list)  # initialized as a empty list

        if hook is not None:
            hook(completed=0, total=num_chunks + has_last_chunk)

        def __append_batch(output, batch_output, **kwargs) -> None:
            output.append(batch_output)
            return

        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks, self.batch_size):
            batch: torch.Tensor = chunks[c : c + self.batch_size]
            B = batch.shape[0]

            batch_embeddings: torch.Tensor = repeat(
                estimated_speaker_embeddings, "ms d -> b ms d", b=B
            )

            batch_outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(batch, batch_embeddings)

            _ = map_with_specifications(
                self.model.specifications, __append_batch, outputs, batch_outputs
            )

            if hook is not None:
                hook(completed=c + self.batch_size, total=num_chunks + has_last_chunk)

        # now len(outputs) = ceil(num_chunks / batch_size)
        # process orphan last chunk
        if has_last_chunk:
            last_embeddings = repeat(estimated_speaker_embeddings, "ms d -> b ms d", b=1)

            last_outputs = self.infer(last_chunk[None], last_embeddings)

            _ = map_with_specifications(
                self.model.specifications, __append_batch, outputs, last_outputs
            )

            if hook is not None:
                hook(
                    completed=num_chunks + has_last_chunk,
                    total=num_chunks + has_last_chunk,
                )
        # now len(outputs) = ceil(num_chunks / batch_size) + 1
        def __vstack(output: List[np.ndarray], **kwargs) -> np.ndarray:
            return np.vstack(output)

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = map_with_specifications(
            self.model.specifications, __vstack, outputs
        )

        def __aggregate(
            outputs: np.ndarray,
            frames: SlidingWindow,
            specifications: Optional[Specifications] = None,
        ) -> SlidingWindowFeature:
            # skip aggregation when requested,
            # or when model outputs just one vector per chunk
            # or when model is permutation-invariant (and not post-processed)
            if (
                self.skip_aggregation  # in DiariZen is set to True
                or specifications.resolution == Resolution.CHUNK
                or (
                    specifications.permutation_invariant
                    and self.pre_aggregation_hook is None
                )
            ):
                frames = SlidingWindow(
                    start=0.0, duration=self.duration, step=self.step
                )
                return SlidingWindowFeature(data=outputs, sliding_window=frames)

            if self.pre_aggregation_hook is not None:
                outputs = self.pre_aggregation_hook(outputs)

            aggregated = self.aggregate(
                SlidingWindowFeature(
                    outputs,
                    SlidingWindow(start=0.0, duration=self.duration, step=self.step),
                ),
                frames,
                warm_up=self.warm_up,
                hamming=True,
                missing=0.0,
                skip_average=False
            )

            # remove padding that was added to last chunk
            if has_last_chunk:
                aggregated.data = aggregated.crop(
                    Segment(0.0, num_samples / sample_rate), mode="loose"
                )

            return aggregated

        return map_with_specifications(
            self.model.specifications, __aggregate, outputs, frames
        )

    def __call__(
        self,
        file: AudioFile,
        session_name: str = None,
        hook: Optional[Callable] = None,
    ) -> Union[
        Tuple[Union[SlidingWindowFeature, np.ndarray]],
        Union[SlidingWindowFeature, np.ndarray],
    ]:
        """Run inference on a whole file

        Parameters
        ----------
        file : AudioFile
            Audio file.
        estimated_speaker_embeddings : Dict[str, Union[np.ndarray, SlidingWindowFeature]]
            Estimated speaker embeddings without pseudo embeddings chosen from training set yet
        hook : callable, optional
            When a callable is provided, it is called everytime a batch is processed
            with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature or np.ndarray
            Model output, as `SlidingWindowFeature` if `window` is set to "sliding"
            and `np.ndarray` if is set to "whole".

        """

        fix_reproducibility(self.device)

        waveform, sample_rate = self.model.audio(file)

        # concat estimated speaker embeddings with pseudo embeddings randomly chosen from training set
        all_spk_list = list(self.ivector_loader.get_speakers_by_session(session_name).keys())
        estimated_session_spk = len(all_spk_list)
        emb_list = list()
        for spk in all_spk_list:
            embeddings = self.ivector_loader.get_speaker_embedding(f"{session_name}-{spk}")
            if embeddings is not None:
                emb_list.append(embeddings)
            else:
                raise NotImplementedError
        emb_data = np.stack(emb_list, axis=0)  # [estimated_session_spk, emb_dim]

        if estimated_session_spk < self.max_speakers:
            pad_speakers = self.max_speakers - estimated_session_spk
            rand_keys = random.sample(self.data_set_speaker, pad_speakers)

            append_spk_list = [self.training_ivector_loader.get_speaker_embedding(k) for k in rand_keys]
            emb_data = torch.from_numpy(np.concatenate([emb_data, np.stack(append_spk_list)], axis=0)).to(torch.float32)

        elif estimated_session_spk > self.max_speakers:
            warnings.warn(
                f"Estimated speakers ({estimated_session_spk}) exceeds max speakers ({self.max_speakers}). "
                f"Selecting first {self.max_speakers} speakers."
            )  # for VoxConverse
            emb_data = torch.from_numpy(emb_data[:self.max_speakers]).to(torch.float32)
        else:
            emb_data = torch.from_numpy(emb_data)

        # perm = torch.arange(self.max_speakers)
        perm = torch.randperm(self.max_speakers)
        emb_data = emb_data[perm]  # randomly shuffle the speaker embeddings

        if self.window == "sliding":
            return (
                self.slide(
                    waveform,
                    sample_rate,
                    emb_data,
                    hook=hook,
                ),
                perm,
                estimated_session_spk
            )

        # TODO: add pseudo speaker embeddings when self.window == "whole"
        outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(waveform[None])

        def __first_sample(outputs: np.ndarray, **kwargs) -> np.ndarray:
            return outputs[0]

        return map_with_specifications(
            self.model.specifications, __first_sample, outputs
        )

    def crop(
        self,
        file: AudioFile,
        chunk: Union[Segment, List[Segment]],
        duration: Optional[float] = None,
        hook: Optional[Callable] = None,
    ) -> Union[
        Tuple[Union[SlidingWindowFeature, np.ndarray]],
        Union[SlidingWindowFeature, np.ndarray],
    ]:
        """Run inference on a chunk or a list of chunks

        Parameters
        ----------
        file : AudioFile
            Audio file.
        chunk : Segment or list of Segment
            Apply model on this chunk. When a list of chunks is provided and
            window is set to "sliding", this is equivalent to calling crop on
            the smallest chunk that contains all chunks. In case window is set
            to "whole", this is equivalent to concatenating each chunk into one
            (artifical) chunk before processing it.
        duration : float, optional
            Enforce chunk duration (in seconds). This is a hack to avoid rounding
            errors that may result in a different number of audio samples for two
            chunks of the same duration.
        hook : callable, optional
            When a callable is provided, it is called everytime a batch is processed
            with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature or np.ndarray
            Model output, as `SlidingWindowFeature` if `window` is set to "sliding"
            and `np.ndarray` if is set to "whole".

        Notes
        -----
        If model needs to be warmed up, remember to extend the requested chunk with the
        corresponding amount of time so that it is actually warmed up when processing the
        chunk of interest:
        >>> chunk_of_interest = Segment(10, 15)
        >>> extended_chunk = Segment(10 - warm_up, 15 + warm_up)
        >>> inference.crop(file, extended_chunk).crop(chunk_of_interest, returns_data=False)
        """

        fix_reproducibility(self.device)

        if self.window == "sliding":
            if not isinstance(chunk, Segment):
                start = min(c.start for c in chunk)
                end = max(c.end for c in chunk)
                chunk = Segment(start=start, end=end)

            waveform, sample_rate = self.model.audio.crop(
                file, chunk, duration=duration
            )
            outputs: Union[
                SlidingWindowFeature, Tuple[SlidingWindowFeature]
            ] = self.slide(waveform, sample_rate, hook=hook)

            def __shift(output: SlidingWindowFeature, **kwargs) -> SlidingWindowFeature:
                frames = output.sliding_window
                shifted_frames = SlidingWindow(
                    start=chunk.start, duration=frames.duration, step=frames.step
                )
                return SlidingWindowFeature(output.data, shifted_frames)

            return map_with_specifications(self.model.specifications, __shift, outputs)

        if isinstance(chunk, Segment):
            waveform, sample_rate = self.model.audio.crop(
                file, chunk, duration=duration
            )
        else:
            waveform = torch.cat(
                [self.model.audio.crop(file, c)[0] for c in chunk], dim=1
            )

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(waveform[None])

        def __first_sample(outputs: np.ndarray, **kwargs) -> np.ndarray:
            return outputs[0]

        return map_with_specifications(
            self.model.specifications, __first_sample, outputs
        )

    @staticmethod
    def aggregate(
        scores: SlidingWindowFeature,  # data, chunk-level SlidingWindow
        frames: SlidingWindow,  # frame-level SlidingWindow
        warm_up: Tuple[float, float] = (0.0, 0.0),
        epsilon: float = 1e-12,
        hamming: bool = False,
        missing: float = np.NaN,
        skip_average: bool = False,
    ) -> SlidingWindowFeature:
        """Aggregation

        Parameters
        ----------
        scores : SlidingWindowFeature
            Raw (unaggregated) scores. Shape is (num_chunks, num_frames_per_chunk, num_classes).
        frames : SlidingWindow
            Frames resolution.
        warm_up : (float, float) tuple, optional
            Left/right warm up duration (in seconds).
        missing : float, optional
            Value used to replace missing (ie all NaNs) values.
        skip_average : bool, optional
            Skip final averaging step.

        Returns
        -------
        aggregated_scores : SlidingWindowFeature
            Aggregated scores. Shape is (num_frames, num_classes)
        """

        num_chunks, num_frames_per_chunk, num_classes = scores.data.shape

        chunks = scores.sliding_window
        frames = SlidingWindow(
            start=chunks.start,
            duration=frames.duration,
            step=frames.step,
        )
        # now, chunks and frames are both pyannote.core.segment.SlidingWindow objects
        masks = 1 - np.isnan(scores)
        scores.data = np.nan_to_num(scores.data, copy=True, nan=0.0)

        # Hamming window used for overlap-add aggregation
        hamming_window = (
            np.hamming(num_frames_per_chunk).reshape(-1, 1)
            if hamming
            else np.ones((num_frames_per_chunk, 1))
        )

        # anything before warm_up_left (and after num_frames_per_chunk - warm_up_right)
        # will not be used in the final aggregation

        # warm-up windows used for overlap-add aggregation
        warm_up_window = np.ones((num_frames_per_chunk, 1))
        # anything before warm_up_left will not contribute to aggregation
        warm_up_left = round(
            warm_up[0] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[:warm_up_left] = epsilon
        # anything after num_frames_per_chunk - warm_up_right either
        warm_up_right = round(
            warm_up[1] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[num_frames_per_chunk - warm_up_right :] = epsilon

        # aggregated_output[i] will be used to store the sum of all predictions
        # for frame #i
        num_frames = (
            frames.closest_frame(
                scores.sliding_window.start
                + scores.sliding_window.duration
                + (num_chunks - 1) * scores.sliding_window.step
                + 0.5 * frames.duration
            )
            + 1
        )
        aggregated_output: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # overlapping_chunk_count[i] will be used to store the number of chunks
        # that contributed to frame #i
        overlapping_chunk_count: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # aggregated_mask[i] will be used to indicate whether
        # at least one non-NAN frame contributed to frame #i
        aggregated_mask: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # loop on the scores of sliding chunks
        for (chunk, score), (_, mask) in zip(scores, masks):
            # chunk ~ Segment
            # score ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray
            # mask ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray

            start_frame = frames.closest_frame(chunk.start + 0.5 * frames.duration)

            aggregated_output[start_frame : start_frame + num_frames_per_chunk] += (
                score * mask * hamming_window * warm_up_window
            )

            overlapping_chunk_count[
                start_frame : start_frame + num_frames_per_chunk
            ] += (mask * hamming_window * warm_up_window)

            aggregated_mask[
                start_frame : start_frame + num_frames_per_chunk
            ] = np.maximum(
                aggregated_mask[start_frame : start_frame + num_frames_per_chunk],
                mask,
            )

        if skip_average:
            average = aggregated_output
        else:
            average = aggregated_output / np.maximum(overlapping_chunk_count, epsilon)

        average[aggregated_mask == 0.0] = missing

        return SlidingWindowFeature(average, frames)

    @staticmethod
    def trim(
        scores: SlidingWindowFeature,
        warm_up: Tuple[float, float] = (0.1, 0.1),
    ) -> SlidingWindowFeature:
        """Trim left and right warm-up regions

        Parameters
        ----------
        scores : SlidingWindowFeature
            (num_chunks, num_frames, num_classes)-shaped scores.
        warm_up : (float, float) tuple
            Left/right warm up ratio of chunk duration.
            Defaults to (0.1, 0.1), i.e. 10% on both sides.

        Returns
        -------
        trimmed : SlidingWindowFeature
            (num_chunks, trimmed_num_frames, num_speakers)-shaped scores
        """

        assert (
            scores.data.ndim == 3  # SlidingWindowFeature
        ), "Inference.trim expects (num_chunks, num_frames, num_classes)-shaped `scores`"
        _, num_frames, _ = scores.data.shape

        chunks = scores.sliding_window

        num_frames_left = round(num_frames * warm_up[0])
        num_frames_right = round(num_frames * warm_up[1])

        num_frames_step = round(num_frames * chunks.step / chunks.duration)
        if num_frames - num_frames_left - num_frames_right < num_frames_step:
            warnings.warn(
                f"Total `warm_up` is so large ({sum(warm_up) * 100:g}% of each chunk) "
                f"that resulting trimmed scores does not cover a whole step ({chunks.step:g}s)"
            )
        new_data = scores.data[:, num_frames_left : num_frames - num_frames_right]

        new_chunks = SlidingWindow(
            start=chunks.start + warm_up[0] * chunks.duration,
            step=chunks.step,
            duration=(1 - warm_up[0] - warm_up[1]) * chunks.duration,
        )

        return SlidingWindowFeature(new_data, new_chunks)
