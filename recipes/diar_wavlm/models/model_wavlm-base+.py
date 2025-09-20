# Copyright 2025 Nanjing University (author: Zeyan Song, clovermaxszy@gmail.com)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[3]))

import math
import numpy as np
from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from pyannote_tsvad.core.model import Model as BaseModel
from models.module.speechbrain_feats import Fbank
from models.module.conformer2 import Conformer
from models.module.transformer_utils import PositionalEncoding
from models.module.wavlm.WavLM import WavLM, WavLMConfig


class Decoder(nn.Module):                           
    def __init__(
        self,
        num_hiddens,
        num_heads,
        ffn_num_hiddens,
        speaker_embedding_dim,
        num_layers,
        dropout,
        ):
        super(Decoder, self).__init__()
        self.spk_proj = nn.Linear(speaker_embedding_dim, num_hiddens)

        self.pos_enc = PositionalEncoding(num_hiddens)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=num_hiddens,
            nhead=num_heads,
            dim_feedforward=ffn_num_hiddens,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.blks = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

    def forward(self, speaker_embeddings, memory):
        decoder_embeddings = self.spk_proj(speaker_embeddings)        
        decoder_embeddings = self.pos_enc(decoder_embeddings)
            
        output = self.blks(
            tgt=decoder_embeddings,
            memory=memory,  # encoder output
        )
        
        return output


class Model(BaseModel):
    def __init__(
        self,
        wavlm_dir: str = None,
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768,
        sample_rate: int = 16000,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        conformer_ffn_hidden: int = 1024,
        conformer_num_head: int = 4,
        conformer_num_layer: int = 4,
        conformer_kernel_size: int = 31,
        conformer_dropout: float = 0.1,
        decoder_num_heads: int = 4,
        decoder_ffn_num_hiddens: int = 1024,
        decoder_layers: int = 4,
        decoder_dropout: float = 0.0,
        max_speakers: int = 10,
        chunk_size: int = 8,  # chunk size in seconds
        num_channels: int = 1,
        selected_channel: int = 0
    ):
        super(Model, self).__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers=max_speakers,
        )

        self.chunk_size = chunk_size
        self.max_speakers = max_speakers
        self.selected_channel = selected_channel

        self.hidden_dim = hidden_dim

        # wavlm 
        self.wavlm_dir = wavlm_dir
        self.wavlm_model, self.wavlm_cfg = self.load_wavlm(wavlm_dir)
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        self.proj = nn.Linear(wavlm_feat_dim, hidden_dim)
        self.lnorm = nn.LayerNorm(hidden_dim)

        self.conformer = Conformer(
            dim=hidden_dim,
            depth=conformer_num_layer,
            heads=conformer_num_head,
            dim_head=hidden_dim // conformer_num_head,
            ff_hidden_size=conformer_ffn_hidden,
            conv_kernel_size=conformer_kernel_size,
            ff_dropout=conformer_dropout,
        )
        self.pos_enc = PositionalEncoding(num_hiddens=hidden_dim)

        self.decoder = Decoder(
            num_hiddens=hidden_dim,
            num_heads=decoder_num_heads,
            ffn_num_hiddens=decoder_ffn_num_hiddens,
            speaker_embedding_dim=embedding_dim,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
            )

        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        decode_frames_per_chunk = self.num_frames(self.chunk_size * sample_rate)
        self.output_layer = nn.Linear(hidden_dim, decode_frames_per_chunk)
        self.activation = self.default_activation()

    def big_lr_parameters(self):
        return [
            *self.weight_sum.parameters(),
            *self.proj.parameters(),
            *self.lnorm.parameters(),
            *self.conformer.parameters(),
            *self.decoder.parameters(),
            *self.gate_mlp.parameters(),
            *self.output_layer.parameters(),
        ]

    def small_lr_parameters(self):
        return [
            *self.wavlm_model.parameters()
        ]

    @property
    def dimension(self) -> int:
        return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """

        return self.wavlm_model.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        return self.wavlm_model.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        return self.wavlm_model.receptive_field_center(frame=frame)

    @property
    def get_rf_info(self, sample_rate=16000):    
        """Return receptive field info to dataset
        """
        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        num_frames = self.num_frames(self.chunk_size * sample_rate)
        duration = receptive_field_size / sample_rate
        step=receptive_field_step / sample_rate
        return num_frames, duration, step

    def load_wavlm(self, wavlm_dir):
        checkpoint = torch.load(wavlm_dir, weights_only=False)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])

        model.encoder.layerdrop = 0.0
        model.feature_grad_mult = 1.0

        return model, cfg

    def wav2wavlm(self, in_wav, model, cfg):
        """
        transform wav to wavlm features
        """
        if cfg.normalize:
            in_wav = torch.nn.functional.layer_norm(in_wav, in_wav.shape[1:])
        rep, layer_results = model.extract_features(in_wav, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        return torch.stack(layer_reps, dim=-1)

    def forward(self, waveforms, embeddings):
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)
        embeddings: (batch, max_speakers, embedding_dim)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        assert waveforms.dim() == 3
        waveforms = waveforms[:, self.selected_channel, :]

        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model, self.wavlm_cfg)
        wavlm_feat = self.weight_sum(wavlm_feat)
        wavlm_feat = torch.squeeze(wavlm_feat, -1)

        encoder_outputs = self.proj(wavlm_feat)
        encoder_outputs = self.lnorm(encoder_outputs)

        encoder_outputs = self.pos_enc(encoder_outputs)
        encoder_outputs = self.conformer(encoder_outputs)

        # Decoder
        decoder_outputs = self.decoder(
            speaker_embeddings=embeddings,
            memory=encoder_outputs
        )

        # logits attractors
        logits_attractors = torch.matmul(encoder_outputs, decoder_outputs.transpose(1, 2))

        # logits decoder linear
        logits_decoder_linear = self.output_layer(decoder_outputs).transpose(1, 2)

        # gating
        g = torch.sigmoid(self.gate_mlp(encoder_outputs))
        logits = g * logits_attractors + (1 - g) * logits_decoder_linear

        outputs = self.activation(logits)

        return outputs  # [B, T, N]


if __name__ == "__main__":
    from thop import profile, clever_format 
    from torchinfo import summary

    device = torch.device('cpu')
    model = Model(
        wavlm_dir="/data/ssd1/zeyan.song/Downloads/SSL/WavLM/WavLM-Base+.pt"
    ).to(device)
    print(model.get_rf_info)
    print(model._receptive_field)  # SlidingWindow object
    print(model.num_frames(16000))
    print(model.receptive_field_size(1))
    print(model.receptive_field_center(0))
    model.eval()

    waveforms = torch.randn(5, 1, 128000).to(device)  # a chunk
    embeddings = torch.randn(5, 10, 256).to(device)
    scores = model(waveforms, embeddings)
    print(scores.shape)  # Should output the shape of the scores tensor

    summary(
        model,
        input_data=[waveforms, embeddings],
        verbose=1,
        col_width=15,
        col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"],  # , "trainable"],
        row_settings=["var_names"],
        depth=3
    )

    flops, params = profile(model, [waveforms, embeddings], verbose=False)
    flops, params = clever_format([flops, params], "%.2f")
    print(f"thop total MACs: {flops}")
    print(f"thop total params: {params}")