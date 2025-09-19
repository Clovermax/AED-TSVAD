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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from pathlib import Path

from enum import Enum

from functools import cached_property
from typing import Dict, List, Optional, Text, Tuple, Union, Sequence

from SDtrainZen.utils import instantiate

from pyannote.core import SlidingWindow

from pyannote.audio.core.io import Audio
from pyannote_tsvad.core.task import (
    Problem,
    Specifications,
)

from pyannote_tsvad.utils.multi_task import map_with_specifications

from torchmetrics import Metric, MetricCollection
from pyannote.audio.torchmetrics import (
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
)

class Resolution(Enum):
    FRAME = 1  # model outputs a sequence of frames
    CHUNK = 2  # model outputs just one vector for the whole chunk


def average_checkpoints(
    # model: nn.Module,
    config,
    checkpoint_list: str,
) -> nn.Module:
    states_dict_list = []
    for ckpt_data in checkpoint_list:
        ckpt_path = ckpt_data['bin_path']
        # copy_model = copy.deepcopy(model)
        copy_model = instantiate(config["model"]["path"], args=config["model"]["args"])
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        copy_model.load_state_dict(checkpoint)
        states_dict_list.append(copy_model.state_dict())
    avg_state_dict = average_states(states_dict_list, torch.device('cpu'))
    # avg_model = copy.deepcopy(model)
    avg_model = instantiate(config["model"]["path"], args=config["model"]["args"])
    avg_model.load_state_dict(avg_state_dict)
    return avg_model

def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key].to(device)
    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state


class Model(nn.Module):
    """ A simple model wrapper to pyannote.audio.core.model
    
    See: https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/core/model.py
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        max_speakers: int = 4,
        duration: int = 5,
        min_duration: int = 5,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        mono: str = "downmix"
    ):
        super().__init__()
        
        if num_channels > 1:
            mono = None

        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.audio = Audio(sample_rate=sample_rate, mono=mono)

        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=duration,
            min_duration=min_duration,
            warm_up=warm_up,
            classes=[f"speaker#{i+1}" for i in range(max_speakers)],
            permutation_invariant=False,
        )

        self.validation_metric = MetricCollection(self.default_metric())
    
    @cached_property
    def _receptive_field(self) -> SlidingWindow:
        """(Internal) frames"""

        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        receptive_field_start = (
            self.receptive_field_center(frame=0) - (receptive_field_size - 1) / 2
        )
        return SlidingWindow(
            start=receptive_field_start / self.sample_rate,
            duration=receptive_field_size / self.sample_rate,
            step=receptive_field_step / self.sample_rate,
        )

    def forward(
        self, waveforms: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        msg = "Class {self.__class__.__name__} should define a `forward` method."
        raise NotImplementedError(msg)

    # convenience function to automate the choice of the final activation function
    def default_activation(self) -> Union[nn.Module, Tuple[nn.Module]]:
        """Guess default activation function according to task specification

            * sigmoid for binary classification
            * log-softmax for regular multi-class classification
            * sigmoid for multi-label classification

        Returns
        -------
        activation : (tuple of) nn.Module
            Activation.
        """

        def __default_activation(
            specifications: Optional[Specifications] = None,
        ) -> nn.Module:
            if specifications.problem == Problem.BINARY_CLASSIFICATION:
                return nn.Sigmoid()

            elif specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:  # Multi-class classification
                return nn.LogSoftmax(dim=-1)

            elif specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
                return nn.Sigmoid()

            else:
                msg = "TODO: implement default activation for other types of problems"
                raise NotImplementedError(msg)

        return map_with_specifications(self.specifications, __default_activation)

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns diarization error rate and its components"""

        return {
            "DiarizationErrorRate": OptimalDiarizationErrorRate(),
            "DiarizationErrorRate/Threshold": OptimalDiarizationErrorRateThreshold(),
            "DiarizationErrorRate/Confusion": OptimalSpeakerConfusionRate(),
            "DiarizationErrorRate/Miss": OptimalMissedDetectionRate(),
            "DiarizationErrorRate/FalseAlarm": OptimalFalseAlarmRate(),
        }

    def __by_name(
        self,
        modules: Union[List[Text], Text],
        recurse: bool = True,
        requires_grad: bool = False,
    ) -> List[Text]:
        """Helper function for freeze_by_name and unfreeze_by_name"""

        updated_modules = list()

        # Force modules to be a list
        if isinstance(modules, str):
            modules = [modules]

        for name in modules:
            module = getattr(self, name)

            for parameter in module.parameters(recurse=True):
                parameter.requires_grad = requires_grad
            module.train(requires_grad)

            # keep track of updated modules
            updated_modules.append(name)

        missing = list(set(modules) - set(updated_modules))
        if missing:
            raise ValueError(f"Could not find the following modules: {missing}.")

        return updated_modules

    def freeze_by_name(
        self,
        modules: Union[Text, List[Text]],
        recurse: bool = True,
    ) -> List[Text]:
        """Freeze modules

        Parameters
        ----------
        modules : list of str, str
            Name(s) of modules to freeze
        recurse : bool, optional
            If True (default), freezes parameters of these modules and all submodules.
            Otherwise, only freezes parameters that are direct members of these modules.

        Returns
        -------
        frozen_modules: list of str
            Names of frozen modules

        Raises
        ------
        ValueError if at least one of `modules` does not exist.
        """

        return self.__by_name(
            modules,
            recurse=recurse,
            requires_grad=False,
        )

    def unfreeze_by_name(
        self,
        modules: Union[List[Text], Text],
        recurse: bool = True,
    ) -> List[Text]:
        """Unfreeze modules

        Parameters
        ----------
        modules : list of str, str
            Name(s) of modules to unfreeze
        recurse : bool, optional
            If True (default), unfreezes parameters of these modules and all submodules.
            Otherwise, only unfreezes parameters that are direct members of these modules.

        Returns
        -------
        unfrozen_modules: list of str
            Names of unfrozen modules

        Raises
        ------
        ValueError if at least one of `modules` does not exist.
        """

        return self.__by_name(modules, recurse=recurse, requires_grad=True)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Union[Path, Text, List],
        config: Union[Path, Text] = None,
    ) -> "Model":
        assert config is not None, "Configuration must be provided to load the model."
        if type(checkpoint) == list:
            print(f'Average model over {len(checkpoint)} checkpoints...')
            print(checkpoint)
            model = average_checkpoints(config, checkpoint)
        else:
            ckpt_loaded = torch.load(checkpoint, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(ckpt_loaded)
        return model

if __name__ == '__main__':
    nnet = Model()