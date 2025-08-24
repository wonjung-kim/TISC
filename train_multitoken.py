#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import pathlib
import warnings
from typing import Optional, Union, Tuple, List

import numpy as np
import PIL
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, AddedToken
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

# from diffusers.utils.import_utils import is_xformers_available

# if is_wandb_available():
#     import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

# from evaluation_function import score_computation
# from auto_initialization import load_clip_resources, precompute_text_features, precompute_text_features_hf, auto_select_tokens, compute_average_image_embeddings
from dataloader_ti import TextualInversionDataset
from embedding import TIMultiTokenEmbedding
from util_functions import log_validation, save_progress
from channel import noisy_prompt_multi_analog_slots_discrete_tokens, wireless_channel
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0.dev0")

logger = get_logger(__name__)


# ------------------ JSONL helpers ------------------
def _read_jsonl(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def _indices_for_category(records, category: str) -> List[int]:
    return sorted({int(r["index"]) for r in records
                   if str(r.get("config")) == str(category) and "index" in r})

def _expand_jindex_arg(jindex_arg, available: List[int]) -> List[int]:
    if jindex_arg is None or str(jindex_arg).lower() == "all":
        return available
    s = str(jindex_arg).strip()
    picks = set()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        picks.update(range(min(a, b), max(a, b) + 1))
    else:
        for tok in s.split(","):
            tok = tok.strip()
            if tok:
                picks.add(int(tok))
    avail = set(available)
    return sorted([p for p in picks if p in avail])


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument(
        "--second_token", type=str, default=None, help="Second token to accelerate training and convergence."
    )
    parser.add_argument(
        "--auto_init_token", action="store_true",
        help="If set, automatically select initializer_token per subfolder via CLIP majority-vote."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="The directory to keep the pre-trained model file"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--test_score",
        action="store_true",
        help="Whether or not to test the similarity scores"
    )
    parser.add_argument(
        "--score_steps",
        type=int,
        default=2000,
        help="Save intermediate similarity scores."
    )
    parser.add_argument(
        "--score_number",
        type=int,
        default=64,
        help="Number of images generated to estimate the clip scores."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--init_coefficients", action="store_true", help="Initialize the coefficient vector from training images."
    )
    parser.add_argument(
        "--num_vectors_per_token",
        type=int,
        default=1,
        help="The number of vectors used to represent the placeholder token.",
    )
    parser.add_argument("--channel_type", default="awgn", type=str, help="awgn/rayleigh")
    parser.add_argument("--snr_db", default=10, type=float)
    parser.add_argument("--jsonl_path", required=True, type=str,
                   help="dream_blip_attn_scr.jsonl 경로")
    parser.add_argument("--jindex", default="all", type=str,
                   help="'all' 또는 '3,5,9' 또는 '2-6' 등")
    parser.add_argument("--json_field", default="starred", choices=["starred", "caption"])

    parser.add_argument("--text_channel_type", type=str, default="none",
        choices=["none","awgn","rayleigh"],
        help="프롬프트(비-placeholder 토큰)용 채널 타입")
    parser.add_argument("--text_snr_db", type=float, default=20.0,
        help="프롬프트 채널 SNR (dB)")
    parser.add_argument("--text_fading", type=str, default="vector",
        choices=["vector","element"], help="프롬프트 채널 페이딩")
    parser.add_argument("--text_per_token", action="store_true", default=True,
        help="토큰별 전력 기준 SNR 정규화")
    parser.add_argument("--text_eval_add_noise", action="store_true", default=True,
        help="eval/validation 시에도 프롬프트 노이즈 적용할지")
    parser.add_argument("--text_freeze_special", action="store_true", default=True,
        help="BOS/EOS/PAD 같은 스페셜 토큰 보호")
    parser.add_argument("--text_keep_placeholder_clean", action="store_true", default=True,
        help="placeholder 토큰은 프롬프트 채널에서 보호")
    parser.add_argument("--text_noise_ratio", type=float, default=1.0,
        help="0~1, 배치 중 이 확률로만 프롬프트 노이즈 적용")
    parser.add_argument("--text_noise_warmup_steps", type=int, default=0,
        help="해당 스텝 이전에는 프롬프트 노이즈 미적용")
 
    parser.add_argument("--text_modulation", type=str, default="adaptive",
        choices=["bpsk","qpsk","16qam","64qam","adaptive"],
        help="프롬프트(비-슬롯) 토큰용 디지털 변조 방식")
    parser.add_argument("--text_target_ber", type=float, default=1e-2,
        help="adaptive 모드에서 목표 BER")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def pick_modulation_by_snr(snr_db: float, target_ber: float = 1e-2, rayleigh: bool = False) -> str:
    """
    아주 간단한 AMC 정책 테이블 (튜닝 권장)
    - AWGN 기준, target_ber~1e-2 근처로 설정한 대략적 문턱
    - Rayleigh면 2~3 dB 정도 보수적으로 문턱 상향
    """
    margin = 2.0 if rayleigh else 0.0
    thr_bpsk_qpsk = 7.0 + margin       # Eb/N0(dB)
    thr_qpsk_16   = 12.0 + margin
    thr_16_64     = 18.0 + margin

    if snr_db < thr_bpsk_qpsk:  return "bpsk"   # 저 SNR
    if snr_db < thr_qpsk_16:    return "qpsk"
    if snr_db < thr_16_64:      return "16qam"
    return "64qam"

def _parse_token_list(x: Optional[str]) -> List[str]:
    if x is None:
        return []
    return [t.strip() for t in str(x).split(",") if t.strip()]

def ensure_single_token_placeholders(tokenizer, text_encoder, literals):
    to_add = []
    for lit in literals:
        tid = tokenizer.convert_tokens_to_ids(lit)
        if not isinstance(tid, int) or tid < 0 or (
            hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id
        ):
            to_add.append(AddedToken(lit, special=True, single_word=True))
    if to_add:
        tokenizer.add_tokens(to_add, special_tokens=True)
        text_encoder.resize_token_embeddings(len(tokenizer))
    return [int(tokenizer.convert_tokens_to_ids(lit)) for lit in literals]

def encode_with_multivector_analog_only_multi(
    *,
    text_encoder,
    tokenizer,
    input_ids: torch.LongTensor,         # [B,S]
    weight_vector: TIMultiTokenEmbedding,
    base_token_ids: List[int],           # 각 placeholder의 "베이스" 토큰 id 리스트 (ex. ["<v1>","<v2>"]의 첫 토큰들)
    attention_mask: Optional[torch.LongTensor] = None,
    channel_type: str = "awgn",
    snr_db: float = 10.0,
    fading: str = "vector",              # 'vector' | 'element'
    slots_only: bool = True,             # True면 placeholder 슬롯 위치에만 채널 적용
    freeze_special: bool = True,
    rng: Optional[torch.Generator] = None,
):
    """
    - TIMultiTokenEmbedding.forward()의 '주입' 로직을 그대로 재현하되,
      self._add_wireless_noise(…)는 쓰지 않고, 주입 후 inputs_embeds에만 채널을 적용.
    - weight_vector.weight: [(P*V), H]를 각 placeholder 그룹의 첫 등장 위치부터 V개 연속으로 주입.
    - slots_only=True 이면 그 주입된 슬롯들만 채널 통과.
    """
    device = input_ids.device
    B, S = input_ids.shape
    E_tok = text_encoder.text_model.embeddings.token_embedding

    inputs_embeds = E_tok(input_ids)                   # [B,S,H]
    H = inputs_embeds.size(-1)
    P = len(base_token_ids)
    V = weight_vector.V                                # vectors per token
    assert weight_vector.weight.shape == (P*V, H), "weight_vector.weight shape mismatch"

    # ── 그룹별 마스크/주입 ─────────────────────────────────────────────
    # inputs_embeds'를 각 그룹(g)에 대해 V개 슬롯에 교체
    slot_union_mask = inputs_embeds.new_zeros((B, S), dtype=torch.bool)
    for g, base_tid in enumerate(base_token_ids):
        ph_mask_g = (input_ids == base_tid)           # [B,S]
        has_g     = ph_mask_g.any(dim=1)              # [B]
        if weight_vector.require_all_placeholders:
            if not torch.all(has_g):
                miss = (~has_g).nonzero(as_tuple=False).flatten().tolist()
                raise ValueError(f"Some rows do not contain placeholder token id {int(base_tid)}. Missing rows: {miss}")

        start_pos_g = ph_mask_g.float().argmax(dim=1)  # [B] (없으면 0이지만 has_g로 거름)
        # M_g: [B,S,V]
        M_g = inputs_embeds.new_zeros((B, S, V))
        for i in range(V):
            idx = start_pos_g + i                     # [B]
            valid = has_g & (idx < S)
            if valid.any():
                b_idx = torch.arange(B, device=device)[valid]
                M_g[b_idx, idx[valid], i] = 1.0

        off = g * V
        vecs_g = weight_vector.weight[off:off+V]      # [V,H]
        replaced_g = torch.einsum("bsv,vh->bsh", M_g, vecs_g)  # [B,S,H]
        keep_mask_g = 1.0 - M_g.sum(dim=2, keepdim=True)      # [B,S,1]
        inputs_embeds = inputs_embeds * keep_mask_g + replaced_g

        slot_union_mask |= (M_g.sum(dim=2) > 0.0)     # [B,S]

    # ── 스페셜 토큰 보호 마스크 ────────────────────────────────────────
    protect_mask = None
    if freeze_special and hasattr(tokenizer, "all_special_ids"):
        sp = torch.tensor(tokenizer.all_special_ids, device=device)
        protect_mask = (input_ids.unsqueeze(-1) == sp).any(dim=-1)  # [B,S]

    # ── 임베딩에만 채널 적용 ───────────────────────────────────────────
    if slots_only:
        # 슬롯 위치만 선택해서 채널 통과 후 scatter
        sel_mask = slot_union_mask
        if protect_mask is not None:
            sel_mask = sel_mask & (~protect_mask)
        if sel_mask.any():
            sel = sel_mask.unsqueeze(-1).expand_as(inputs_embeds)    # [B,S,H]
            x_sel = inputs_embeds[sel].view(-1, H)                   # [Nslot,H]
            y_sel = wireless_channel(
                x_sel, channel_type=channel_type, snr_db=snr_db,
                complex_mode="real", per_sample=True,
                training=True, eval_add_noise=True,
                fading=fading, rng=rng
            )
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[sel] = y_sel.reshape(-1)
    else:
        y = wireless_channel(
            inputs_embeds, channel_type=channel_type, snr_db=snr_db,
            complex_mode="real", per_sample=True,
            training=True, eval_add_noise=True,
            fading=fading, rng=rng
        )
        if protect_mask is not None and protect_mask.any():
            keep = protect_mask.unsqueeze(-1).expand_as(y)
            inputs_embeds = torch.where(keep, inputs_embeds, y)
        else:
            inputs_embeds = y

    # ── CLIP 인코더 통과 (pos emb + encoder + LN) ─────────────────────
    if attention_mask is None:
        attention_mask = torch.ones((B, S), device=device, dtype=torch.long)

    pos_ids = text_encoder.text_model.embeddings.position_ids[:, :S]
    pos_emb = text_encoder.text_model.embeddings.position_embedding(pos_ids)
    hidden  = inputs_embeds + pos_emb

    causal = _create_4d_causal_attention_mask((B, S), hidden.dtype, hidden.device)
    enc_out = text_encoder.text_model.encoder(
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        causal_attention_mask=causal,
        return_dict=True
    )
    last_hidden = text_encoder.text_model.final_layer_norm(enc_out.last_hidden_state)  # [B,S,H]
    return last_hidden

@torch.no_grad()
def bake_validation_embeddings_per_placeholder(
    text_encoder,
    tokenizer,
    base_placeholders,          # 예: ["<v1>", "<v2>"]
    learned_W,                  # torch.Tensor, shape = [(P*V), H]
    num_vectors_per_token: int, # V
    compose: str = "mean",      # "mean" | "first" | "normed-mean"
):
    # 1) 단일 토큰 보장 + id 얻기
    base_ids = ensure_single_token_placeholders(tokenizer, text_encoder, base_placeholders)

    # 2) 그룹별 합성 후 해당 베이스 토큰 embedding 교체
    emb = text_encoder.get_input_embeddings().weight
    off = 0
    for tid in base_ids:
        vecs = learned_W[off : off + num_vectors_per_token]  # [V, H]
        if compose == "mean":
            new_vec = vecs.mean(dim=0)
        elif compose == "first":
            new_vec = vecs[0]
        elif compose == "normed-mean":
            new_vec = F.normalize(vecs, dim=-1).mean(dim=0)
            new_vec = F.normalize(new_vec, dim=-1)
        else:
            raise ValueError(f"Unknown compose='{compose}'")
        emb[tid].copy_(new_vec.to(emb.dtype))
        off += num_vectors_per_token

def main():
    args = parse_args()

    # 콤마로 분리된 다중 토큰 파싱
    base_placeholders = _parse_token_list(args.placeholder_token)  # 예: "<v1>,<v2>"
    if len(base_placeholders) == 0:
        raise ValueError("At least one placeholder token is required (comma-separated).")

    base_initializers = _parse_token_list(args.initializer_token)  # 예: "object,style" 또는 "object"
    if len(base_initializers) not in (1, len(base_placeholders)):
        raise ValueError(
            f"--initializer_token must be 1 item (broadcast) or match #placeholders. "
            f"Got {len(base_initializers)} for {len(base_placeholders)} placeholders."
    )
    # ───────────────────────────────────────────────────────────────────
    # (1) 원래 args.train_data_dir와 args.output_dir를 저장해 둡니다.
    parent_train_dir = args.train_data_dir
    parent_output_dir = args.output_dir

    # ───────────────────────────────────────────────────────────────────
    # (2) parent_train_dir 내부의 하위 폴더 리스트를 구해서, 없으면 self‐dataset으로 처리
    entries = os.listdir(parent_train_dir)
    subdirs = [d for d in entries if os.path.isdir(os.path.join(parent_train_dir, d))]
    if subdirs:
        # A-1, A-2, ... 과 같이 하위폴더마다 학습
        dataset_dirs  = [os.path.join(parent_train_dir, d) for d in subdirs]
        dataset_names = subdirs
    else:
        # 내부에 폴더가 없으면 parent_train_dir 자체를 한 건으로 처리
        dataset_dirs  = [parent_train_dir]
        # 출력 디렉터리명으로 쓸 단일 이름은 디렉터리의 basename
        dataset_names = [os.path.basename(parent_train_dir)]

    # JSONL 읽기 (카테고리별 jindex 조회에 사용)
    records = _read_jsonl(args.jsonl_path)
    train_flag=False
    # ───────────────────────────────────────────────────────────────────
    # (3) 각각의 dataset_dirs, dataset_names 쌍으로 학습 루프
    for sub_dir, category_name in zip(dataset_dirs, dataset_names):
        args.train_data_dir = sub_dir
        if category_name == "grey_sloth_plushie": train_flag=True
        if train_flag == False: continue
        available = _indices_for_category(records, category_name)
        if not available:
            print(f"[WARN] JSONL에 category='{category_name}'의 index 항목이 없습니다. skip.")
            continue

        jindex_list = _expand_jindex_arg(args.jindex, available)
        if not jindex_list:
            print(f"[WARN] --jindex={args.jindex} 가 category='{category_name}'에 대해 비었습니다. skip.")
            continue

        for j in jindex_list:
            args.output_dir = os.path.join(parent_output_dir, category_name, f"img{j:01d}")
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"[INFO] category={category_name}  jindex={j}  -> {args.output_dir}")

            # (3-3) Accelerator, 로거 등 초기화 (하위 폴더마다 독립 실행)
            accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
            accelerator = Accelerator(
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                mixed_precision=args.mixed_precision,
                log_with=args.report_to,
                project_dir=args.output_dir,
                project_config=accelerator_project_config,
            )

            if args.report_to == "wandb":
                if not is_wandb_available():
                    raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
            )
            logger.info(accelerator.state, main_process_only=False)
            if accelerator.is_local_main_process:
                transformers.utils.logging.set_verbosity_warning()
                diffusers.utils.logging.set_verbosity_info()
            else:
                transformers.utils.logging.set_verbosity_error()
                diffusers.utils.logging.set_verbosity_error()

            # 시드 설정
            if args.seed is not None:
                set_seed(args.seed)

            # ───────────────────────────────────────────────────────────────────
            # (3-4) 모델 및 토크나이저 로드 직후의 placeholder 관련 부분 전체 교체
            if args.tokenizer_name:
                tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
            else:
                tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

            noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
            vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
            unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

            # ── [B-1] 다중 placeholder를 '토큰 그룹'으로 확장
            # 예) base_placeholders=["<v1>","<v2>"], num_vectors_per_token=3
            #     → placeholder_tokens=["<v1>","<v1>-1","<v1>-2","<v2>","<v2>-1","<v2>-2"]
            placeholder_tokens: List[str] = []
            placeholder_groups: List[List[str]] = []  # 각 플레이스홀더의 마이크로 토큰 묶음

            for base in base_placeholders:
                group = [base] + [f"{base}-{i}" for i in range(1, args.num_vectors_per_token)]
                placeholder_groups.append(group)
                placeholder_tokens.extend(group)

            # 토크나이저 등록
            num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
            if num_added_tokens != len(placeholder_tokens):
                raise ValueError(
                    f"Some of placeholder tokens already exist in tokenizer: {placeholder_tokens}"
                )

            # ID 해석
            placeholder_id_groups: List[List[int]] = [
                tokenizer.convert_tokens_to_ids(group) for group in placeholder_groups
            ]
            base_token_ids: List[int] = [g[0] for g in placeholder_id_groups]

            # ── [B-2] 이니셜라이저 ID 준비 (1개면 broadcast)
            if len(base_initializers) == 1:
                base_initializers = base_initializers * len(base_placeholders)
            init_ids: List[int] = [
                tokenizer.convert_tokens_to_ids(tok) for tok in base_initializers
            ]
            if any((i is None) or (i == tokenizer.unk_token_id) for i in init_ids):
                raise ValueError(f"Initializer tokens must be single-vocab tokens: {base_initializers}")

            # ── [B-3] 텍스트 인코더 임베딩 테이블에서 초기화
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data

            with torch.no_grad():
                for group_ids, init_id in zip(placeholder_id_groups, init_ids):
                    for tid in group_ids:
                        token_embeds[tid] = token_embeds[init_id]

            # ── [B-4] TITokenEmbedding 초기화 (다중 플레이스홀더 + 다중벡터 대응)
            # init_coeff: [총_벡터_개수, D]
            D = token_embeds.shape[1]
            initial_embeds_multi = []
            for init_id in init_ids:
                vec = token_embeds[init_id].clone()  # [D]
                initial_embeds_multi.extend([vec] * args.num_vectors_per_token)
            initial_embeds_multi = torch.stack(initial_embeds_multi, dim=0)  # [(P*V), D]

            weight_vector = TIMultiTokenEmbedding(
                args,
                init_coeff=initial_embeds_multi,         # [(P*V), D]
                placeholder_base_ids=base_token_ids,     # [P]
                # num_vectors_per_token=args.num_vectors_per_token,
            )

            # with torch.no_grad():
            #     for token_id in placeholder_token_ids:
            #         token_embeds[token_id] = token_embeds[initializer_token_id]

            # 필요한 파라미터들 freeze
            vae.requires_grad_(False)
            unet.requires_grad_(False)
            text_encoder.text_model.encoder.requires_grad_(False)
            text_encoder.text_model.final_layer_norm.requires_grad_(False)
            text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
            text_encoder.text_model.embeddings.token_embedding.requires_grad_(False)

            if args.gradient_checkpointing:
                unet.train()
                text_encoder.gradient_checkpointing_enable()
                unet.enable_gradient_checkpointing()

            if args.allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True

            if args.scale_lr:
                args.learning_rate = (
                    args.learning_rate
                    * args.gradient_accumulation_steps
                    * args.train_batch_size
                    * accelerator.num_processes
                )

            # ───────────────────────────────────────────────────────────────────
            # (3-7) Optimizer / Scheduler 초기화
            optimizer = torch.optim.AdamW(
                weight_vector.parameters(),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

            train_dataset = TextualInversionDataset(
                data_root=sub_dir,
                tokenizer=tokenizer,
                size=args.resolution,
                placeholder_token=args.placeholder_token,
                repeats=args.repeats,
                jsonl_path=args.jsonl_path,
                json_category=category_name,
                json_index=j,
                json_prompt_field=args.json_field,
                single_image_idx=j,     
                center_crop=args.center_crop,
                set="train",
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.dataloader_num_workers,
            )

            if args.validation_epochs is not None:
                warnings.warn(
                    f"FutureWarning: You specified validation_epochs={args.validation_epochs}. "
                    "This is deprecated in favor of validation_steps."
                    f" Setting validation_steps = {args.validation_epochs * len(train_dataset)}",
                    FutureWarning,
                )
                args.validation_steps = args.validation_epochs * len(train_dataset)

            overrode_max_train_steps = False
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            if args.max_train_steps is None:
                args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
                overrode_max_train_steps = True

            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            )

            # ───────────────────────────────────────────────────────────────────
            # (3-8) accelerator.prepare() 호출 (model, optimizer, dataloader, scheduler, weight_vector, candidate_embedding_matrix, vocab_ids)
            text_encoder, optimizer, train_dataloader, lr_scheduler, weight_vector = accelerator.prepare(
                text_encoder,
                optimizer,
                train_dataloader,
                lr_scheduler,
                weight_vector
            )

            if accelerator.is_main_process:
                weights = accelerator.unwrap_model(weight_vector).weight.detach().cpu().squeeze().tolist()

            weight_dtype = torch.float32
            if accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16

            unet.to(accelerator.device, dtype=weight_dtype)
            vae.to(accelerator.device, dtype=weight_dtype)

            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            if overrode_max_train_steps:
                args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

            if accelerator.is_main_process:
                accelerator.init_trackers(f"textual_inversion_{category_name}", config=vars(args))

            # ── noisy prompt 저장용 버퍼 ──────────────────────────────────────
            last_noisy_prompts = None   # 마지막으로 사용된 noisy prompt들의 리스트[str]
            last_noisy_meta = None      # 마지막 noisy 배치의 메타정보
            ever_text_noise = False     # 학습 중 한 번이라도 text noise 사용 여부

            # ───────────────────────────────────────────────────────────────────
            # (3-9) 실제 학습 루프
            total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
            logger.info("***** Running training *****")
            logger.info(f"  Subfolder = {category_name}")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {args.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {args.max_train_steps}")
            logger.info(f"  Initializer token = {args.initializer_token}")
            logger.info(f"  BLIP-guided text prompt = {train_dataset.fixed_text_prompt}")

            global_step = 0
            first_epoch = 0
            if args.resume_from_checkpoint:
                if args.resume_from_checkpoint != "latest":
                    path = os.path.basename(args.resume_from_checkpoint)
                else:
                    dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = dirs[-1] if len(dirs) > 0 else None

                if path is None:
                    accelerator.print(
                        f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                    )
                    args.resume_from_checkpoint = None
                else:
                    accelerator.print(f"Resuming from checkpoint {path}")
                    accelerator.load_state(os.path.join(args.output_dir, path))
                    global_step = int(path.split("-")[1])
                    resume_global_step = global_step * args.gradient_accumulation_steps
                    first_epoch = global_step // num_update_steps_per_epoch
                    resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

            progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Steps [{category_name}]")

            if args.test_score:
                score_i2i_dict = {}
                score_i2t_dict = {}
                score_i2i_dino_dict = {}
            else:
                score_i2i_dict = None
                score_i2t_dict = None
                score_i2i_dino_dict = None

            for epoch in range(first_epoch, args.num_train_epochs):
                weight_vector.train()
                for step, batch in enumerate(train_dataloader):
                    # Resume logic
                    if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                        if step % args.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                        continue

                    with accelerator.accumulate(weight_vector):
                        # 1) 이미지 → latent
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                        latents = latents * vae.config.scaling_factor

                        # 2) 노이즈 샘플링
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                        ).long()
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


                        # --- NEW: 프롬프트(비-슬롯) 디지털 채널(QAM) 적용 ---
                        use_text_noise = (
                            args.text_modulation is not None and
                            args.text_modulation != "" and
                            args.text_channel_type != "none" and
                            global_step >= args.text_noise_warmup_steps and
                            (random.random() < args.text_noise_ratio)
                        )

                        if use_text_noise:
                            emb_module = text_encoder.get_input_embeddings()
                            B = batch["input_ids"].shape[0]
                            L = tokenizer.model_max_length
                            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                            base_prompt = train_dataset.fixed_text_prompt


                            noisy_ids_list = []
                            batch_noisy_texts = []     # ★ 추가: 이 배치에서 실제로 사용된 noisy prompt 문자열들
                            batch_mods = []            # ★ 추가: 선택된 변조 방식 기록(Adaptive일 때 확인용)

                            for b in range(B):
                                # 샘플별 독립 노이즈를 위해 seed 또는 per-sample generator 사용
                                if args.seed is not None:
                                    seed_val = int(args.seed + global_step * 100000 + b)
                                    rng = torch.Generator(device=accelerator.device).manual_seed(seed_val)
                                else:
                                    rng = None

                                # adaptive 선택이면 SNR 기반 변조 고르기
                                if args.text_modulation == "adaptive":
                                    mod = pick_modulation_by_snr(
                                        args.text_snr_db,
                                        target_ber=args.text_target_ber,
                                        rayleigh=(args.text_channel_type == "rayleigh"),
                                    )
                                else:
                                    mod = args.text_modulation

                                # ★ 하이브리드 채널 적용: 슬롯=아날로그(여기서는 무시), 비-슬롯=디지털 QAM
                                # 멀티 플레이스홀더 리스트
                                slot_token_literals = base_placeholders  # 예: ["<v1>", "<v2>", ...]
                                # (선택) 각 리터럴에 대한 임베딩을 명시하고 싶으면 아래처럼 제공 가능
                                slot_embed_map = {lit: emb_module.weight[tid] for lit, tid in zip(base_placeholders, base_token_ids)}

                                res = noisy_prompt_multi_analog_slots_discrete_tokens(
                                    base_prompt,
                                    tokenizer=tokenizer,
                                    embedding_module=emb_module,
                                    slot_token_literals=slot_token_literals,
                                    slot_embed_map=slot_embed_map,
                                    channel_type=args.text_channel_type,      # 'awgn' or 'rayleigh'
                                    snr_db=args.text_snr_db,                  # Eb/N0 (dB)
                                    fading="vector",
                                    modulation=mod,                           # 'bpsk'|'qpsk'|'16qam'|'64qam'
                                    freeze_special=args.text_freeze_special,
                                    # (선택) 디코딩 결과에서 각 리터럴을 무엇으로 보일지
                                    render_placeholder_map={lit: lit for lit in slot_token_literals},
                                    rng=rng,
                                )

                                ids = res["recovered_input_ids"]      # [1, T_rec] (예: [1,12])
                                text_str = res["recovered_text"]          # ★ noisy prompt 문자열
                                batch_noisy_texts.append(text_str)        # ★ 수집
                                batch_mods.append(mod)                    # ★ 수집

                                # ▶ pad/truncate to [1, L] (L=77)
                                T_rec = ids.shape[1]
                                if T_rec < L:
                                    pad = torch.full((1, L - T_rec), pad_id, device=ids.device, dtype=ids.dtype)
                                    ids = torch.cat([ids, pad], dim=1)
                                elif T_rec > L:
                                    ids = ids[:, :L]

                                noisy_ids_list.append(ids)

                            noisy_input_ids = torch.cat(noisy_ids_list, dim=0)      # [B,T]

                            # placeholder 보호: 혹시라도 변조 과정에서 바뀌었으면 원본으로 되돌림
                            for pid in base_token_ids:  # ← (기존: placeholder_token_ids 에서 첫 원소만 쓰던 것을 교체)
                                noisy_input_ids = torch.where(
                                    batch["input_ids"] == pid,
                                    batch["input_ids"], noisy_input_ids
                                )
                            
                            # ★ 마지막 noisy 배치 정보로 갱신
                            last_noisy_prompts = batch_noisy_texts
                            last_noisy_meta = {
                                "global_step": int(global_step),
                                "channel_type": str(args.text_channel_type),
                                "snr_db": float(args.text_snr_db),
                                "modulations": batch_mods,  # 배치 내 샘플별로 기록
                                "placeholder_token": str(args.placeholder_token),
                            }
                            ever_text_noise = True

                        else:
                            print("Not use prompt noise system")
                            noisy_input_ids = batch["input_ids"]
                        # --- NEW 끝 ---



                        # # 3) 텍스트 임베딩 (placeholder_token에 weight_vector 적용)
                        # encoder_hidden_states = encode_with_multivector_analog_only_multi(
                        #                         text_encoder=accelerator.unwrap_model(text_encoder),
                        #                         tokenizer=tokenizer,
                        #                         input_ids=noisy_input_ids,                  # 디지털 프롬프트 노이즈를 쓰지 않으면 batch["input_ids"]
                        #                         weight_vector=weight_vector,                # TIMultiTokenEmbedding
                        #                         base_token_ids=base_token_ids,              # 위에서 만든 각 placeholder의 베이스 토큰 id 리스트
                        #                         attention_mask=batch.get("attention_mask", None),
                        #                         channel_type=args.channel_type,             # --channel_type (awgn/rayleigh)
                        #                         snr_db=args.snr_db,                         # --snr_db
                        #                         fading="vector",                            # 필요시 "element"
                        #                         slots_only=True,                            # 슬롯 위치에만 노이즈 주기
                        #                         freeze_special=True,
                        #                         rng=(torch.Generator(device=accelerator.device).manual_seed(
                        #                                 int(args.seed + global_step*100000) if args.seed is not None else 0
                        #                             ) if args.seed is not None else None),
                        #                     ).to(dtype=weight_dtype)
                        encoder_hidden_states = weight_vector(text_encoder, tokenizer, noisy_input_ids)[0].to(dtype=weight_dtype)

                        # 4) 노이즈 예측
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                        # 5) loss target 결정
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()


                    # ───────────────────────────────────────────────────────────────────
                    # (3-10) optimizer.step 이 발생한 순간 (accelerator.sync_gradients 시)
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        # ───────────────────────────────────────────────────────────────────
                        # (3-11) learned_embeds 저장 (args.save_steps마다)
                        if global_step % args.save_steps == 0:
                            save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}")
                            save_progress(weight_vector, accelerator, args, save_path)

                        # ───────────────────────────────────────────────────────────────────
                        # (3-12) checkpoint 저장 (args.checkpointing_steps마다)
                        if global_step % args.checkpointing_steps == 0:
                            if accelerator.is_main_process:
                                state_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(state_path)
                                logger.info(f"[{category_name}] Saved state to {state_path}")

                        # ───────────────────────────────────────────────────────────────────
                        # (3-13) validation (args.validation_steps마다)
                        if global_step % args.validation_steps == 0:
                            if accelerator.is_main_process:
                                weights_model = accelerator.unwrap_model(weight_vector)
                            
                            # ── 추가: validation용 placeholder 임베딩 갱신 ──
                            with torch.no_grad():
                                q_w = accelerator.unwrap_model(weight_vector).weight.to(accelerator.device)  # [(P*V), H]

                            # base_placeholders = ["<v1>", "<v2>", ...]  (학습 시작 시 파싱했던 그 리스트)
                            bake_validation_embeddings_per_placeholder(
                                text_encoder,
                                tokenizer,
                                base_placeholders=base_placeholders,
                                learned_W=q_w,
                                num_vectors_per_token=args.num_vectors_per_token,
                                compose="mean",  # 또는 "first", "normed-mean"
                            )

                            accelerator.wait_for_everyone()   # 다중-GPU일 때 동기화
                            # ───────────────────────────────────────────────────────────────────
                            # 깨끗한 프롬프트
                            clean_prompt = train_dataset.fixed_text_prompt
                            log_validation(
                                logger, text_encoder, tokenizer, unet, vae,
                                args, accelerator, weight_dtype, global_step, clean_prompt, "clean"
                            )

                            # (옵션) 노이즈 프롬프트
                            if args.text_channel_type != "none" and args.text_eval_add_noise:
                                with torch.no_grad():
                                    emb_module = text_encoder.get_input_embeddings()
                                    # 모듈레이션 선택(한 번)
                                    if args.text_modulation == "adaptive":
                                        mod = pick_modulation_by_snr(
                                            args.text_snr_db,
                                            target_ber=args.text_target_ber,
                                            rayleigh=(args.text_channel_type == "rayleigh"),
                                        )
                                    else:
                                        mod = args.text_modulation

                                    rng = torch.Generator(device=accelerator.device).manual_seed(0)

                                    noisy_out = noisy_prompt_multi_analog_slots_discrete_tokens(
                                        clean_prompt,
                                        tokenizer=tokenizer,
                                        embedding_module=emb_module,
                                        slot_token_literals=base_placeholders,
                                        slot_embed_map=None,
                                        channel_type=args.text_channel_type,
                                        snr_db=args.text_snr_db,
                                        fading="vector",
                                        modulation=mod,
                                        freeze_special=args.text_freeze_special,
                                        render_placeholder_map={lit: lit for lit in base_placeholders},
                                        rng=rng,
                                    )
                                    noisy_prompt = noisy_out["recovered_text"]

                                    # placeholder 토큰 문자열 보호: (ex) "<v*>"가 깨졌다면 원상 복구
                                    if args.text_keep_placeholder_clean and args.placeholder_token not in noisy_prompt:
                                        # 아주 보수적으로, 원래 placeholder 토큰이 사라졌으면 다시 붙여줌
                                        # (필요시 더 정교하게 처리 가능)
                                        noisy_prompt = clean_prompt

                                log_validation(
                                    logger, text_encoder, tokenizer, unet, vae,
                                    args, accelerator, weight_dtype, global_step, noisy_prompt, "noisy"
                                )
                            
                        # ───────────────────────────────────────────────────────────────────
                        # (3-14) test_score 절차
                        # if args.test_score and global_step % args.score_steps == 0:
                            # score_i2i_dict, score_i2t_dict, score_i2i_dino_dict = score_computation(
                            #     text_encoder,
                            #     tokenizer,
                            #     unet,
                            #     vae,
                            #     args,
                            #     accelerator,
                            #     weight_dtype,
                            #     global_step,
                            #     logger,
                            #     score_i2i_dict,
                            #     score_i2t_dict,
                            #     score_i2i_dino_dict,
                            # )

                    # ───────────────────────────────────────────────────────────────────
                    # (3-15) progress_bar 로그 업데이트
                    qstr = f"#Vec={args.num_vectors_per_token}"
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "Q": qstr}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if global_step >= args.max_train_steps:
                        break

                if global_step >= args.max_train_steps:
                    break

            # ───────────────────────────────────────────────────────────────────
            # (3-16) 학습 끝난 뒤, pipeline 저장 처리
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # full model 저장 여부 결정
                if args.push_to_hub and args.only_save_embeds:
                    logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
                    save_full_model = True
                else:
                    save_full_model = not args.only_save_embeds

                if save_full_model:
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        vae=vae,
                        unet=unet,
                        tokenizer=tokenizer,
                    )
                    os.makedirs(args.model_dir, exist_ok=True)
                    pipeline.save_pretrained(args.model_dir)

                # embed 파일 저장
                if not args.test_score:
                    save_path = os.path.join(args.output_dir, "learned_embeds")
                    save_progress(weight_vector, accelerator, args, save_path)

                # initializer token 저장
                init_word_path = os.path.join(args.output_dir, "initialization_word.txt")
                with open(init_word_path, "w", encoding="utf-8") as f:
                    f.write(f"{args.initializer_token}")
                
                # noisy prompt 저장 (학습 중 text noise가 한 번이라도 적용된 경우)
                if ever_text_noise and last_noisy_prompts is not None:
                    noisy_dump = {
                        "base_prompt": train_dataset.fixed_text_prompt,  # 원본 고정 프롬프트
                        "last_batch_noisy_prompts": last_noisy_prompts,  # 마지막 배치에서 실제로 투입된 noisy 프롬프트들
                        "meta": last_noisy_meta,                          # 채널/변조/스텝 등
                    }
                    noisy_path = os.path.join(args.output_dir, "noisy_training_prompts.json")
                    with open(noisy_path, "w", encoding="utf-8") as f:
                        json.dump(noisy_dump, f, ensure_ascii=False, indent=2)
                    logger.info(f"[{category_name}] Saved noisy training prompts → {noisy_path}")

                # test_score 결과물 저장
                if args.test_score:
                    test_score_path = os.path.join(args.output_dir, "test_score")
                    os.makedirs(test_score_path, exist_ok=True)
                    with open(os.path.join(test_score_path, "clip_i2i_score.txt"), "w", encoding="utf-8") as f:
                        f.write(json.dumps(score_i2i_dict))
                    with open(os.path.join(test_score_path, "clip_i2t_score.txt"), "w", encoding="utf-8") as f:
                        f.write(json.dumps(score_i2t_dict))
                    with open(os.path.join(test_score_path, "dino_i2i_score.txt"), "w", encoding="utf-8") as f:
                        f.write(json.dumps(score_i2i_dino_dict))


            accelerator.end_training()
        # ───────────────────────────────────────────────────────────────────
        # (3-17) 다음 하위 폴더를 위해 loop가 계속됨
        print(f"=== Subfolder [{category_name}] training finished. Results saved to {args.output_dir} ===\n\n")

    # ───────────────────────────────────────────────────────────────────
    # (4) 모든 하위 폴더 루프가 종료되면 main() 종료
    print("All subfolder trainings are complete.")



if __name__ == "__main__":
    main()
