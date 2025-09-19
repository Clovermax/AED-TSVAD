# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright 2025 Nanjing University (author: Zeyan Song, clovermaxszy@gmail.com)

"""Speaker diarization pipelines"""

import functools
import itertools
import math
import textwrap
import warnings
from typing import Dict, Any, Callable, Optional, Text, Union
from tqdm import tqdm
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from einops import rearrange
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.pipeline.parameter import ParamDict, Uniform

from pyannote.audio import Audio, Pipeline
from pyannote_tsvad.core.model import Model
from pyannote_tsvad.core.inference_ivector import Inference
from pyannote_tsvad.pipelines.utils.diarization_ivector import SpeakerDiarizationMixin
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.audio.utils.signal import binarize


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def load_scp(scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

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

class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    segmentation: Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation@2022.07".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    segmentation_step: float, optional
        The segmentation model is applied on a window sliding over the whole audio file.
        `segmentation_step` controls the step of this window, provided as a ratio of its
        duration. Defaults to 0.1 (i.e. 90% overlap between two consecuive windows).
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/embedding@2022.07".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    embedding_exclude_overlap : bool, optional
        Exclude overlapping speech regions when extracting embeddings.
        Defaults (False) to use the whole speech.
    clustering : str, optional
        Clustering algorithm. See pyannote.audio.pipelines.clustering.Clustering
        for available options. Defaults to "AgglomerativeClustering".
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 1.
    embedding_batch_size : int, optional
        Batch size used for speaker embedding. Defaults to 1.
    der_variant : dict, optional
        Optimize for a variant of diarization error rate.
        Defaults to {"collar": 0.0, "skip_overlap": False}. This is used in `get_metric`
        when instantiating the metric: GreedyDiarizationErrorRate(**der_variant).
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Usage
    -----
    # perform (unconstrained) diarization
    >>> diarization = pipeline("/path/to/audio.wav")

    # perform diarization, targetting exactly 4 speakers
    >>> diarization = pipeline("/path/to/audio.wav", num_speakers=4)

    # perform diarization, with at least 2 speakers and at most 10 speakers
    >>> diarization = pipeline("/path/to/audio.wav", min_speakers=2, max_speakers=10)

    # perform diarization and get one representative embedding per speaker
    >>> diarization, embeddings = pipeline("/path/to/audio.wav", return_embeddings=True)
    >>> for s, speaker in enumerate(diarization.labels()):
    ...     # embeddings[s] is the embedding of speaker `speaker`

    Hyper-parameters
    ----------------
    segmentation.threshold
    segmentation.min_duration_off
    clustering.???
    """

    def __init__(
        self,
        config: Union[Dict, Any] = None,
        segmentation: PipelineModel = "pyannote/segmentation@2022.07",
        segmentation_duration: float = 8.0,
        segmentation_step: float = 0.1,
        segmentation_batch_size: int = 1,
        der_variant: Optional[dict] = None,
        device: Optional[torch.device] = torch.device("cuda"),
        max_speakers: int = 8,
        speaker_embedding_txt: Optional[Text] = None,
        training_speaker_embedding_txt: Optional[Text] = None,
    ):
        super().__init__()

        self.segmentation_model = segmentation
        model: Model = Model.from_pretrained(segmentation, config=config)

        self.segmentation_step = segmentation_step

        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}
        self.device = device
        self.max_speakers = max_speakers
        self.segmentation_duration = segmentation_duration

        self._segmentation = Inference(
            model,
            duration=self.segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
            batch_size=segmentation_batch_size,
            device=device,
            speaker_embedding_txt=speaker_embedding_txt,
            training_speaker_embedding_txt=training_speaker_embedding_txt,
            max_speakers=self.max_speakers,
        )

        self.segmentation = ParamDict(
            threshold=Uniform(0.1, 0.9),
            min_duration_off=Uniform(0.0, 1.0),
        )

    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    def default_parameters(self):
        raise NotImplementedError()

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

    @property
    def CACHED_SEGMENTATION(self):
        return "training_cache/segmentation"

    def get_segmentations(
        self,
        file,
        session_name: str,
        hook=None,
    ) -> SlidingWindowFeature:
        """Apply segmentation model

        Parameter
        ---------
        file : AudioFile
        session_name: str
        hook : Optional[Callable]

        Returns
        -------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        """

        if hook is not None:
            hook = functools.partial(hook, "segmentation", None)

        segmentations, perm, estimated_sess_spk = self._segmentation(
            file,
            session_name=session_name,
            hook=hook,
        )
        return segmentations, perm, estimated_sess_spk

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        session_speaker_permutation: torch.Tensor,  # map valid speaker indices from shuffled "valid + pesudo" speaker indices
        estimated_session_speaker_count: int,  # number of valid speaker embeddings
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build final discrete diarization for TS-VAD: delete pseudo speaker labels

        Parameters
        ----------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Raw speaker segmentation.
        count : (total_num_frames, 1) SlidingWindowFeature
            Instantaneous number of active speakers.

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        num_chunks, num_frames, max_speakers = segmentations.data.shape
        assert max_speakers == self.max_speakers

        inv_perm = torch.argsort(session_speaker_permutation)
        filtered_segmentations = segmentations.data[:, :, inv_perm][:, :, :estimated_session_speaker_count]

        filtered_segmentations = SlidingWindowFeature(
            filtered_segmentations, segmentations.sliding_window
        )

        return self.to_diarization(filtered_segmentations, count)
