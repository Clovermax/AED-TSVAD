# AED-TSVAD

Pytorch implementation of "Attention-Based Encoder-Decoder Target-Speaker Voice Activity Detection for Robust Speaker Diarization" submitted to ICASSP 2026

## Pretrained checkpoints

Pretrained checkpoints, together with training logs and evaluation results are available [here](https://drive.google.com/drive/folders/1xVZNnkix5mDpIqsfZEFbmqpmoKQgelxV).

## Data split

Detailed data split is also available [here](https://drive.google.com/drive/folders/1xVZNnkix5mDpIqsfZEFbmqpmoKQgelxV), you should put the `data` folder under `recipes/diar_wavlm`

Single-channel recordings are used as released. For multichannel data, we use the first channel (for CHiME-6, the first channel of the first array). Official train/dev/eval splits are adopted when available. For AISHELL-4 which lacks an official development set, we follow the split used in DiariZen[1]. For datasets providing only development and evaluation sets (e.g., DIHARD and VoxConverse), we follow the split in SSND[2] by using the first 80\% of the original development set for training and the remaining 20\% for validation. For NOTSOFAR-SC, we use single-channel sessions from training sets 1 and 2 for training and dev set 1 for validation.

## Training

Example:

```bash
cd recipes/diar_wavlm/RUN_TRAIN
bash run_training.sh --stage 1 --stop_stage 1 --gpus 0,1,2,3 --dataset Compound --conf_name model_wavlm-base+_rvector_cs-ft --use_dual_opt true

```

## Inference

Some path need change before running inference.

```bash
cd recipes/diar_wavlm/RUN_EVAL
bash run_eval_rvector.sh --exp_name model_wavlm-base+_rvector_cs-ft --gpu_id 0 --init_rttm_type diarizen_base_s80_pretrained_vbx --feature_extractor_type wavlm-base+

```

## Todo

- [ ] Inference single WAV
- [ ] Make the inference pipeline easier to use
- [ ] Check reproducibility
- [ ] Add kaldi-based i-vector extraction scripts (You can also refer to NSD-MS2S)

I will update as soon as possible if the paper is accepted.

## References

[1] Han, J., et al. (2024). "Leveraging Self-Supervised Learning for Speaker Diarization." arXiv preprint arXiv:2409.09408.
[2] Cheng, M., et al. (2024). "Sequence-to-Sequence Neural Diarization with Automatic Speaker Detection and Representation." arXiv preprint arXiv:2411.13849.

## Acknowledgements

We would like to thank the developers and maintainers of the following open-source projects that our work builds upon or was inspired by:
[Pyannote](https://github.com/pyannote/pyannote-audio)
[DiariZen](https://github.com/BUTSpeechFIT/DiariZen)
[NSD-MS2S](https://github.com/liyunlongaaa/NSD-MS2S)
