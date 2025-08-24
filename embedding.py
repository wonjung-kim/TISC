import math
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from typing import Optional, Union, Tuple, List

class TITokenEmbedding(nn.Module):
    def __init__(self, args, init_coeff: torch.Tensor):
        super().__init__()

        # --- 무선 채널 시뮬레이션 옵션 ---
        self.channel_type = args.channel_type   # 'awgn' 또는 'rayleigh'
        self.snr_db = args.snr_db               # dB 스케일의 SNR

        if init_coeff is None:
            raise ValueError("init_coeff must be provided for TI-only mode")

        # init_coeff: [H] 또는 [N, H]
        if init_coeff.dim() == 1:
            H = init_coeff.shape[0]
            w_init = init_coeff.unsqueeze(0).clone().detach()  # [1, H]
            self.num_vectors = 1
        elif init_coeff.dim() == 2:
            self.num_vectors, H = init_coeff.shape
            w_init = init_coeff.clone().detach()               # [N, H]
        else:
            raise ValueError(f"init_coeff must be 1D or 2D, got shape {tuple(init_coeff.shape)}")

        # args.num_vectors_per_token 이 주어졌다면 모양 검증
        if hasattr(args, "num_vectors_per_token") and args.num_vectors_per_token is not None:
            assert self.num_vectors == int(args.num_vectors_per_token), \
                f"Expected {args.num_vectors_per_token} vectors, but got {self.num_vectors}"

        self.weight = nn.Parameter(w_init, requires_grad=True)

    def _add_wireless_noise(self, x: torch.Tensor):
        """ y = h*x + n (학습 중에만 노이즈 추가) """
        if not self.training:
            return x

        signal_power = torch.mean(torch.pow(x, 2))
        if signal_power == 0:
            return x

        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear

        if self.channel_type == "awgn":
            h = 1.0
        elif self.channel_type == "rayleigh":
            h_real = torch.randn_like(x) / math.sqrt(2)
            h_imag = torch.randn_like(x) / math.sqrt(2)
            h = torch.sqrt(h_real**2 + h_imag**2)
        else:
            raise ValueError("channel_type must be 'awgn' or 'rayleigh'")

        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * noise_std
        return h * x + noise

    def forward(self, text_encoder, input_ids,
                placeholder_token_id):

        # 1) wireless noise
        new_embed_2d = self._add_wireless_noise(self.weight)  # [N, H]
        N = new_embed_2d.shape[0]

        # 2) 텍스트 임베딩 (detach 금지)
        inputs_embeds = text_encoder.text_model.embeddings.token_embedding(input_ids)  # [B, S, H]
        B, S, H = inputs_embeds.shape

        # 3) 배치별 placeholder 시작 위치 찾기 (각 배치에 1회 등장 가정)
        ph_mask = (input_ids == placeholder_token_id)  # [B, S]
        if not ph_mask.any():
            raise ValueError("Placeholder token not found in input_ids")

        has_ph = ph_mask.any(dim=1)
        if not torch.all(has_ph):
            raise ValueError("Some rows do not contain the placeholder token")

        start_pos = ph_mask.float().argmax(dim=1)  # [B]

        # 4) [B, S, N] 마스크 생성 (N개 연속 토큰 교체)
        M = inputs_embeds.new_zeros((B, S, N))
        for i in range(N):
            idx = start_pos + i              # [B]
            valid = idx < S
            if valid.any():
                b_idx = torch.arange(B, device=idx.device)[valid]
                M[b_idx, idx[valid], i] = 1.0

        # 5) 교체 텐서 합성: [B,S,H]
        replaced_part = torch.einsum('bsn,nh->bsh', M, new_embed_2d)  # self.weight로 grad 전파
        keep_mask = 1.0 - M.sum(dim=2, keepdim=True)                  # [B,S,1]
        inputs_embeds = inputs_embeds * keep_mask + replaced_part

        # 6) 이후 CLIP 경로 동일
        seq_len = input_ids.shape[-1]
        pos_ids = text_encoder.text_model.embeddings.position_ids[:, :seq_len]
        hidden = inputs_embeds + text_encoder.text_model.embeddings.position_embedding(pos_ids)

        mask = _create_4d_causal_attention_mask(
            (hidden.size(0), seq_len), hidden.dtype, hidden.device
        )

        enc_out = text_encoder.text_model.encoder(
            inputs_embeds=hidden,
            causal_attention_mask=mask,
            return_dict=True
        )

        last_hidden = text_encoder.text_model.final_layer_norm(enc_out.last_hidden_state)
        pooled = last_hidden[
            torch.arange(last_hidden.size(0), device=hidden.device),
            input_ids.argmax(dim=-1)  # 기존 코드 유지
        ]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden, pooler_output=pooled)

    # --- 편의 함수 ---
    def get_noisy_weights(self):
        return self._add_wireless_noise(self.weight)



class TIMultiTokenEmbedding(nn.Module):
    """
    Textual Inversion (TI) – Multi-Placeholder version.

    - Supports P ≥ 1 placeholder *tokens* (each can have V ≥ 1 learnable vectors).
    - For each placeholder base token id, we replace V consecutive token embeddings
      starting at the first occurrence position of that base token in the sequence.
      (Same behavior as TITokenEmbedding, generalized to multiple placeholders.)

    Shapes:
      * init_coeff:
          - [H]                    -> P=1, V=1
          - [V, H]                 -> P=1, V=V
          - [P*V, H]               -> P=P, V=V
      * self.weight: [(P*V), H]
    """
    def __init__(
        self,
        args,
        init_coeff: torch.Tensor,
        placeholder_base_ids: List[int],
        num_vectors_per_token: Optional[int] = None,
        require_all_placeholders: bool = True,
    ):
        super().__init__()

        # --- Wireless channel sim options (kept for compatibility) ---
        self.channel_type = getattr(args, "channel_type", "awgn")   # 'awgn' or 'rayleigh'
        self.snr_db = float(getattr(args, "snr_db", 10.0))          # dB

        # --- Placeholder config ---
        if not isinstance(placeholder_base_ids, list) or len(placeholder_base_ids) == 0:
            raise ValueError("placeholder_base_ids must be a non-empty List[int].")
        self.placeholder_base_ids = placeholder_base_ids  # length P
        self.P = len(self.placeholder_base_ids)

        # Decide V (vectors per placeholder)
        if num_vectors_per_token is None:
            # fall back to args if present
            num_vectors_per_token = int(getattr(args, "num_vectors_per_token", 1))
        self.V = int(num_vectors_per_token)
        if self.V <= 0:
            raise ValueError("num_vectors_per_token (V) must be >= 1.")

        self.require_all_placeholders = bool(require_all_placeholders)

        # --- Normalize init_coeff to [(P*V), H] and register parameter ---
        if init_coeff is None:
            raise ValueError("init_coeff must be provided.")

        if init_coeff.dim() == 1:
            # [H] -> P=1, V=1
            if self.P != 1 or self.V != 1:
                raise ValueError(
                    f"init_coeff is [H] but P={self.P}, V={self.V}. "
                    f"Provide [P*V, H] or [V, H] to match."
                )
            H = init_coeff.shape[0]
            w_init = init_coeff.unsqueeze(0).clone().detach()  # [1, H]

        elif init_coeff.dim() == 2:
            N_or_V, H = init_coeff.shape
            # Case A: provided for a single placeholder [V,H] and P must be 1
            if N_or_V == self.V and self.P == 1:
                w_init = init_coeff.clone().detach()  # [V, H]
            # Case B: provided flattened for all placeholders [(P*V), H]
            elif N_or_V == self.P * self.V:
                w_init = init_coeff.clone().detach()  # [(P*V), H]
            else:
                raise ValueError(
                    f"init_coeff shape {tuple(init_coeff.shape)} incompatible with P={self.P}, V={self.V}. "
                    f"Expected [V,H] (when P=1) or [(P*V),H]."
                )
        else:
            raise ValueError(f"init_coeff must be 1D or 2D, got shape {tuple(init_coeff.shape)}")

        # Validate against args.num_vectors_per_token if present
        if hasattr(args, "num_vectors_per_token") and args.num_vectors_per_token is not None:
            exp_V = int(args.num_vectors_per_token)
            if self.V != exp_V:
                raise AssertionError(
                    f"Expected num_vectors_per_token={exp_V}, but got V={self.V}"
                )

        # Register learnable weights
        self.weight = nn.Parameter(w_init, requires_grad=True)  # [(P*V), H]
        self.H = self.weight.shape[1]

    # ------------------------ Channel noise ------------------------
    def _add_wireless_noise(self, x: torch.Tensor) -> torch.Tensor:
        """ y = h*x + n (noise only during training) """
        if not self.training:
            return x

        signal_power = torch.mean(x.pow(2))
        if signal_power.item() == 0.0:
            return x

        snr_linear = 10.0 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear

        if self.channel_type == "awgn":
            h = 1.0
        elif self.channel_type == "rayleigh":
            # Magnitude of complex Gaussian
            h_real = torch.randn_like(x) / math.sqrt(2.0)
            h_imag = torch.randn_like(x) / math.sqrt(2.0)
            h = torch.sqrt(h_real**2 + h_imag**2)
        else:
            raise ValueError("channel_type must be 'awgn' or 'rayleigh'")

        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * noise_std
        return h * x + noise

    # ------------------------ Forward ------------------------
    def forward(
        self,
        text_encoder,
        tokenizer,
        input_ids: torch.LongTensor,
    ):
        """
        Replaces, for each placeholder base token id, V consecutive token embeddings
        starting at the FIRST occurrence position per sample.

        Args:
          - text_encoder: CLIPTextModel
          - input_ids: [B, S]
        Returns:
          BaseModelOutputWithPooling(last_hidden_state=[B,S,H], pooler_output=[B,H])
        """
        # (1) prepare noisy vectors: [(P*V), H]
        new_embed = self._add_wireless_noise(self.weight)
        B, S = input_ids.shape

        # (2) lookup original token embeddings: [B,S,H]
        inputs_embeds = text_encoder.text_model.embeddings.token_embedding(input_ids)  # [B,S,H]

        # (3) for each placeholder group g, replace V slots
        for g, base_tid in enumerate(self.placeholder_base_ids):
            # mask for base token in each row: [B,S]
            ph_mask_g = (input_ids == base_tid)

            if self.require_all_placeholders:
                if not ph_mask_g.any():
                    raise ValueError(
                        f"Placeholder token id {int(base_tid)} not found in input_ids."
                    )
                has_g = ph_mask_g.any(dim=1)
                if not torch.all(has_g):
                    miss = (~has_g).nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(
                        f"Some rows do not contain placeholder token id {int(base_tid)}. Missing rows: {miss}"
                    )

            # start positions (first occurrence) per sample: [B]
            # If a row doesn't have it and require_all_placeholders=False, we skip that row via 'valid' later.
            start_pos_g = ph_mask_g.float().argmax(dim=1)  # if row all 0, argmax=0; we'll gate by 'has'
            has_g = ph_mask_g.any(dim=1)

            # build mask M_g: [B,S,V]
            M_g = inputs_embeds.new_zeros((B, S, self.V))
            for i in range(self.V):
                idx = start_pos_g + i  # [B]
                valid = has_g & (idx < S)
                if valid.any():
                    b_idx = torch.arange(B, device=idx.device)[valid]
                    M_g[b_idx, idx[valid], i] = 1.0

            # slice group's vectors: [V,H]
            off = g * self.V
            vecs_g = new_embed[off:off + self.V]  # [V,H]

            # replacement for this group: [B,S,H]
            replaced_g = torch.einsum('bsn,nh->bsh', M_g, vecs_g)

            # zero-out the original positions for this group and add replaced
            keep_mask_g = 1.0 - M_g.sum(dim=2, keepdim=True)  # [B,S,1]
            inputs_embeds = inputs_embeds * keep_mask_g + replaced_g

        # (4) downstream CLIP blocks (same as original)
        seq_len = input_ids.shape[-1]
        pos_ids = text_encoder.text_model.embeddings.position_ids[:, :seq_len]
        hidden = inputs_embeds + text_encoder.text_model.embeddings.position_embedding(pos_ids)

        mask = _create_4d_causal_attention_mask(
            (hidden.size(0), seq_len), hidden.dtype, hidden.device
        )

        enc_out = text_encoder.text_model.encoder(
            inputs_embeds=hidden,
            causal_attention_mask=mask,
            return_dict=True
        )

        last_hidden = text_encoder.text_model.final_layer_norm(enc_out.last_hidden_state)

        # Pooling rule kept identical to your TITokenEmbedding:
        eos_id = tokenizer.eos_token_id
        eos_pos = (input_ids == eos_id).float().argmax(dim=1)  # [B]
        pooled = last_hidden[
            torch.arange(last_hidden.size(0), device=hidden.device),
            eos_pos
        ]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden,
            pooler_output=pooled
        )

    # ------------------------ Utils ------------------------
    def get_noisy_weights(self) -> torch.Tensor:
        """Return current weights with wireless noise applied (training-time)."""
        return self._add_wireless_noise(self.weight)