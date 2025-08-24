import math
from typing import List, Optional, Dict, Union, Tuple
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, CLIPTokenizer

# ─────────────────────────────────────────────────────────────────────
#                      ── 공 통   유 틸 ──
# ─────────────────────────────────────────────────────────────────────

def _rng_randn(shape, *, dtype, device, rng: Optional[torch.Generator]):
    return torch.randn(shape, dtype=dtype, device=device, generator=rng)

def _rng_randn_cplx(shape, *, device, rng: Optional[torch.Generator]):
    nr = torch.randn(shape, dtype=torch.float32, device=device, generator=rng)
    ni = torch.randn(shape, dtype=torch.float32, device=device, generator=rng)
    return torch.complex(nr, ni)

def _normalize_token_literal(tok: str) -> str:
    # BPE/SentencePiece 선행 공백 마커 제거 후 비교
    return tok.lstrip("Ġ▁")

# -------- 비트/그레이 코드 유틸 --------
def _int_to_bits(x: torch.LongTensor, L: int, *, device) -> torch.LongTensor:
    x = x.unsqueeze(-1)
    shifts = torch.arange(L - 1, -1, -1, device=device, dtype=torch.long)
    return (x >> shifts) & 1

def _bits_to_int(bits: torch.LongTensor, *, device) -> torch.LongTensor:
    L = bits.shape[-1]
    shifts = torch.arange(L - 1, -1, -1, device=device, dtype=torch.long)
    return (bits * (1 << shifts)).sum(dim=-1)

def _binary_to_gray(b: torch.LongTensor) -> torch.LongTensor:
    return b ^ (b >> 1)

def _gray_bits_to_binary_bits(gray_bits: torch.LongTensor) -> torch.LongTensor:
    # 누적 XOR (mod 2)
    csum = torch.cumsum(gray_bits, dim=-1)
    return torch.remainder(csum, 2)

# -------- 변조(QAM) 유틸 --------
def _pam_levels(m: int, *, device) -> torch.Tensor:
    M = 1 << m
    return torch.arange(-(M - 1), M, 2, device=device, dtype=torch.float32)  # odd sequence

def _mod_params(mod: str, *, device) -> Tuple[int, float, Optional[torch.Tensor], int]:
    mod = mod.lower()
    if mod == "bpsk":
        return 1, 1.0, None, 0
    if mod == "qpsk":
        return 2, 1.0 / math.sqrt(2.0), _pam_levels(1, device=device), 1
    if mod == "16qam":
        return 4, 1.0 / math.sqrt(10.0), _pam_levels(2, device=device), 2
    if mod == "64qam":
        return 6, 1.0 / math.sqrt(42.0), _pam_levels(3, device=device), 3
    raise ValueError("Unsupported modulation. Use 'bpsk'|'qpsk'|'16qam'|'64qam'.")

def _qam_modulate(bits_sym: torch.LongTensor, mod: str, *, device) -> torch.Tensor:
    k, scale, pam, m_dim = _mod_params(mod, device=device)
    if mod == "bpsk":
        s = 2.0 * bits_sym.squeeze(-1).to(torch.float32) - 1.0
        return (s + 0.0j).to(torch.complex64)

    bI = _bits_to_int(bits_sym[:, :m_dim], device=device)
    bQ = _bits_to_int(bits_sym[:, m_dim:], device=device)
    gI = _binary_to_gray(bI)
    gQ = _binary_to_gray(bQ)
    aI = pam[gI]; aQ = pam[gQ]
    return (scale * (aI + 1j * aQ)).to(torch.complex64)

def _qam_demodulate(y_eq: torch.Tensor, mod: str, *, device) -> torch.LongTensor:
    k, scale, pam, m_dim = _mod_params(mod, device=device)
    if mod == "bpsk":
        return (y_eq.real >= 0).to(torch.long).unsqueeze(-1)

    rI = (y_eq.real / scale).to(torch.float32)
    rQ = (y_eq.imag / scale).to(torch.float32)
    pam_vec = pam.view(1, -1)
    dI = torch.abs(rI.view(-1, 1) - pam_vec)
    dQ = torch.abs(rQ.view(-1, 1) - pam_vec)
    gI_hat = torch.argmin(dI, dim=-1)
    gQ_hat = torch.argmin(dQ, dim=-1)
    gI_bits = _int_to_bits(gI_hat, m_dim, device=device)
    gQ_bits = _int_to_bits(gQ_hat, m_dim, device=device)
    bI_bits = _gray_bits_to_binary_bits(gI_bits)
    bQ_bits = _gray_bits_to_binary_bits(gQ_bits)
    return torch.cat([bI_bits, bQ_bits], dim=-1)

# -------- 채널/노이즈 유틸 --------
def _snr_lin_from_db(db: float) -> float:
    return 10.0 ** (db / 10.0)

def _make_fading(shape, *, channel_type: str, is_complex: bool, like_dtype, device, rng):
    if channel_type == "awgn":
        if is_complex:
            dt = torch.complex64 if like_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128
            return torch.ones(shape, dtype=dt, device=device)
        return torch.ones(shape, dtype=like_dtype, device=device)

    # Rayleigh
    if is_complex:
        hr = _rng_randn(shape, dtype=torch.float32, device=device, rng=rng) / math.sqrt(2.0)
        hi = _rng_randn(shape, dtype=torch.float32, device=device, rng=rng) / math.sqrt(2.0)
        return torch.complex(hr, hi)  # E[|h|^2]=1
    else:
        hr = _rng_randn(shape, dtype=like_dtype, device=device, rng=rng) / math.sqrt(2.0)
        hi = _rng_randn(shape, dtype=like_dtype, device=device, rng=rng) / math.sqrt(2.0)
        return torch.sqrt(hr.pow(2) + hi.pow(2))

def _fading_shape_for(x_like: torch.Tensor, fading: str) -> Tuple[int, ...]:
    if fading == "element":
        return x_like.shape
    # vector(블록): 마지막 차원만 1
    if x_like.ndim >= 2:
        return (*x_like.shape[:-1], 1)
    return x_like.shape

def _signal_power(x: torch.Tensor, *, per_sample: bool) -> Union[torch.Tensor, torch.Tensor]:
    p = x.pow(2)
    if per_sample and p.ndim >= 2:
        return p.mean(dim=-1, keepdim=True)  # [M,1] or [B,T,1]
    return p.mean()

def _nearest_vocab_decode(
    noisy_embeds: torch.Tensor,
    input_ids: torch.LongTensor,
    tokenizer: PreTrainedTokenizer,
    embedding_module,
    *,
    freeze_special: bool,
    restrict_vocab_to: Optional[List[int]] = None,
) -> Tuple[torch.LongTensor, List[str]]:
    W = embedding_module.weight.detach()
    W32 = F.normalize(W.float(), dim=-1)        # [V,D]
    Y = F.normalize(noisy_embeds.float(), dim=-1)  # [B,T,D]
    device = noisy_embeds.device

    if restrict_vocab_to:
        idx_map = torch.tensor(restrict_vocab_to, device=device, dtype=torch.long)
        W32_sub = W32.index_select(0, idx_map)
        sims = torch.einsum("btd,vd->btv", Y, W32_sub)
        topk = sims.argmax(dim=-1)
        decoded_ids = idx_map[topk]
    else:
        sims = torch.einsum("btd,vd->btv", Y, W32)
        decoded_ids = sims.argmax(dim=-1)

    if freeze_special and hasattr(tokenizer, "all_special_ids"):
        special = torch.tensor(tokenizer.all_special_ids, device=device)
        is_special = (input_ids.unsqueeze(-1) == special).any(dim=-1)  # [B,T]
        decoded_ids = torch.where(is_special, input_ids, decoded_ids)

    decoded_text = tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
    return decoded_ids, decoded_text

def _apply_analog_on_vectors(
    vectors: torch.Tensor,
    *,
    snr_db: float,
    channel_type: str,
    fading: str,
    rng: Optional[torch.Generator]
) -> torch.Tensor:
    """vectors: [..., D] 실수 임베딩. per-vector 전력 기준으로 AWGN/Rayleigh 적용."""
    device, dtype = vectors.device, vectors.dtype
    snr_lin = _snr_lin_from_db(snr_db)
    sp = vectors.pow(2).mean(dim=-1, keepdim=True)
    nstd = torch.sqrt(torch.clamp(sp / snr_lin, min=0.0))  # [...,1]

    h_shape = _fading_shape_for(vectors, fading)
    h = _make_fading(h_shape, channel_type=channel_type, is_complex=False, like_dtype=dtype, device=device, rng=rng)
    n = _rng_randn(vectors.shape, dtype=dtype, device=device, rng=rng) * nstd
    return h * vectors + n

def _digital_path_ids_through_qam(
    ids: torch.LongTensor,
    *,
    digi_mask: torch.Tensor,
    vocab_size: int,
    bits_per_token: int,
    channel_type: str,
    snr_db: float,             # Eb/N0 (dB)
    modulation: str,
    rng: Optional[torch.Generator]
) -> torch.LongTensor:
    """디지털 경로(QAM)로 통과한 토큰 ID 복구."""
    device = ids.device
    if not digi_mask.any():
        return ids

    ids_digi = ids[digi_mask]                   # [Ntok]
    Ntok = ids_digi.numel()
    k, _, _, _ = _mod_params(modulation, device=device)

    # Eb/N0 -> Es/N0
    ebn0_lin = _snr_lin_from_db(snr_db)
    esn0_lin = ebn0_lin * k
    N0 = 1.0 / max(esn0_lin, 1e-12)

    # 비트 패킹
    L = bits_per_token
    bits = _int_to_bits(ids_digi, L, device=device)
    Ns = (L + k - 1) // k
    pad = Ns * k - L
    if pad > 0:
        bits_pack = torch.cat(
            [bits, torch.zeros((Ntok, pad), device=device, dtype=torch.long)], dim=-1
        )
    else:
        bits_pack = bits
    bits_sym = bits_pack.view(Ntok * Ns, k)     # [Nsym, k]

    # 변조 → 채널 → 등화 → 복호
    s = _qam_modulate(bits_sym, modulation, device=device)  # [Nsym]
    if channel_type == "awgn":
        h = torch.ones_like(s)
    elif channel_type == "rayleigh":
        h = _rng_randn_cplx(s.shape, device=device, rng=rng) / math.sqrt(2.0)
    else:
        raise ValueError("channel_type must be 'awgn' or 'rayleigh'")

    noise = math.sqrt(N0 / 2.0) * _rng_randn_cplx(s.shape, device=device, rng=rng)
    y = h * s + noise
    y_eq = y / h

    bits_sym_hat = _qam_demodulate(y_eq, modulation, device=device)  # [Nsym,k]
    bits_pack_hat = bits_sym_hat.view(Ntok, Ns * k)[:, :L]
    ids_hat = _bits_to_int(bits_pack_hat, device=device)
    ids_hat = torch.remainder(ids_hat, vocab_size)
    ids[digi_mask] = ids_hat
    return ids

# ─────────────────────────────────────────────────────────────────────
#                  ── 원래 제공된 공개 API들 ──
#          (시그니처/출력 구조 변경 없이 내부만 간소화)
# ─────────────────────────────────────────────────────────────────────

def wireless_channel(
    x: torch.Tensor,
    *,
    channel_type: str = "awgn",      # 'awgn' | 'rayleigh'
    snr_db: float = 20.0,
    complex_mode: str = "pair",      # 'pair' | 'real' | 'complex'
    per_sample: bool = True,         # True면 각 행별 전력로 노이즈 스케일링
    training: bool = True,           # 모델의 self.training 전달
    eval_add_noise: bool = True,    # 평가 시에도 노이즈 추가할지
    fading: str = "vector",         # 'element' | 'vector' (블록 페이딩)
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    if (not training) and (not eval_add_noise):
        return x
    if channel_type not in ("awgn", "rayleigh"):
        raise ValueError("channel_type must be 'awgn' or 'rayleigh'")
    if complex_mode not in ("pair", "real", "complex"):
        raise ValueError("complex_mode must be 'pair', 'real', or 'complex'")
    if fading not in ("element", "vector"):
        raise ValueError("fading must be 'element' or 'vector'")

    device = x.device
    snr_lin = _snr_lin_from_db(snr_db)

    # 복소 경로 구성
    def _as_complex_from_pairs(xr: torch.Tensor) -> torch.Tensor:
        if xr.shape[-1] % 2 != 0:
            raise ValueError("Last dim must be even to pair (I,Q).")
        i = xr[..., 0::2]; q = xr[..., 1::2]
        return torch.complex(i, q)

    def _as_pairs_from_complex(z: torch.Tensor, last_dim_size: int) -> torch.Tensor:
        y = torch.stack((z.real, z.imag), dim=-1)
        return y.reshape(*z.shape[:-1], last_dim_size)

    if complex_mode == "complex":
        if not torch.is_complex(x):
            raise ValueError("complex_mode='complex' expects complex input.")
        xc, xr = x, None
    elif complex_mode == "pair":
        xc, xr = _as_complex_from_pairs(x), None
    else:
        xc, xr = None, x

    if xc is not None:
        # 복소 경로
        h_shape = x.shape if fading == "element" else (*x.shape[:-1], 1)
        h = _make_fading(h_shape[:-1] if fading == "vector" else h_shape,  # align complex shape
                         channel_type=channel_type, is_complex=True,
                         like_dtype=x.real.dtype if x.is_complex() else torch.float32,
                         device=device, rng=rng)
        # Complex AWGN: var(real)=var(imag)=npow/2
        sp = (xc.real.pow(2) + xc.imag.pow(2))
        if per_sample and sp.ndim >= 2:
            sp = sp.mean(dim=-1, keepdim=True)
        else:
            sp = sp.mean()
        npow = sp / snr_lin
        if torch.is_tensor(npow):
            std = torch.sqrt(torch.clamp(npow, min=0.0) / 2.0)
            while std.ndim < xc.ndim:
                std = std.unsqueeze(-1)
        else:
            std = math.sqrt(max(npow, 0.0) / 2.0)
        n = torch.complex(
            _rng_randn(xc.shape, dtype=torch.float32, device=device, rng=rng) * std,
            _rng_randn(xc.shape, dtype=torch.float32, device=device, rng=rng) * std
        )
        y = h * xc + n
        return _as_pairs_from_complex(y, x.shape[-1]) if complex_mode == "pair" else y

    # 실수 경로
    h_shape = _fading_shape_for(xr, fading)  # type: ignore[arg-type]
    h = _make_fading(h_shape, channel_type=channel_type, is_complex=False,
                     like_dtype=xr.dtype, device=device, rng=rng)  # type: ignore[arg-type]
    sp = _signal_power(xr, per_sample=per_sample)  # type: ignore[arg-type]
    npow = sp / snr_lin
    if torch.is_tensor(npow):
        nstd = torch.sqrt(torch.clamp(npow, min=0.0))
        while nstd.ndim < xr.ndim:  # type: ignore[union-attr]
            nstd = nstd.unsqueeze(-1)
    else:
        nstd = math.sqrt(max(npow, 0.0))
    n = _rng_randn(xr.shape, dtype=xr.dtype, device=device, rng=rng) * nstd  # type: ignore[union-attr]
    y = h * xr + n                                                           # type: ignore[operator]
    if torch.is_tensor(sp) and per_sample and y.ndim >= 2:
        zero_mask = (sp <= 1e-12).expand_as(y)
        y = torch.where(zero_mask, xr, y)                                    # type: ignore[arg-type]
    return y

def text_through_channel(
    prompts_or_ids: Union[List[str], torch.LongTensor],
    *,
    tokenizer: PreTrainedTokenizer,
    embedding_module,                 # e.g., model.get_input_embeddings()
    channel_type: str = "awgn",       # 'awgn' | 'rayleigh'
    snr_db: float = 20.0,
    fading: str = "vector",           # 'element' | 'vector'
    per_token: bool = True,
    training: bool = True,
    eval_add_noise: bool = True,
    freeze_special: bool = True,
    decode_strategy: str = "nearest", # only 'nearest' supported
    restrict_vocab_to: Optional[List[int]] = None,
    return_inputs_embeds: bool = True,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, Union[torch.LongTensor, List[str], torch.Tensor]]:
    device = embedding_module.weight.device
    dtype  = embedding_module.weight.dtype
    snr_lin = _snr_lin_from_db(snr_db)

    # 0) ids/attention
    if isinstance(prompts_or_ids, list):
        enc = tokenizer(prompts_or_ids, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
    else:
        input_ids = prompts_or_ids.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)

    # 1) 임베딩
    E = embedding_module
    embeds = E(input_ids).to(dtype)  # [B,T,D]
    B, T, D = embeds.shape

    if (not training) and (not eval_add_noise):
        noisy_embeds = embeds
    else:
        # per-token 전력
        sp = embeds.pow(2).mean(dim=-1, keepdim=True) if per_token else embeds.pow(2).mean()
        npow = sp / snr_lin

        # 페이딩
        if channel_type not in ("awgn", "rayleigh"):
            raise ValueError("channel_type must be 'awgn' or 'rayleigh'.")
        if fading not in ("element", "vector"):
            raise ValueError("fading must be 'element' or 'vector'.")
        h_shape = embeds.shape if fading == "element" else (B, T, 1)
        h = _make_fading(h_shape, channel_type=channel_type, is_complex=False,
                         like_dtype=dtype, device=device, rng=rng)

        # AWGN
        if isinstance(npow, torch.Tensor):
            nstd = torch.sqrt(torch.clamp(npow, min=0.0))
            if nstd.ndim < embeds.ndim:
                nstd = nstd.expand_as(embeds)
        else:
            nstd = math.sqrt(max(npow, 0.0))
        n = _rng_randn(embeds.shape, dtype=dtype, device=device, rng=rng) * nstd

        noisy_embeds = h * embeds + n
        if per_token and isinstance(sp, torch.Tensor):
            zero_mask = (sp <= 1e-12).expand_as(noisy_embeds)
            noisy_embeds = torch.where(zero_mask, embeds, noisy_embeds)

    out: Dict[str, Union[torch.LongTensor, List[str], torch.Tensor]] = {}

    if decode_strategy == "nearest":
        decoded_ids, decoded_text = _nearest_vocab_decode(
            noisy_embeds, input_ids, tokenizer, embedding_module,
            freeze_special=freeze_special, restrict_vocab_to=restrict_vocab_to
        )
        out["noisy_input_ids"] = decoded_ids
        out["noisy_text"] = decoded_text

    if return_inputs_embeds:
        out["inputs_embeds"] = noisy_embeds
        out["attention_mask"] = attention_mask
    return out

@torch.no_grad()
def noisy_prompt_analog_slot_discrete_tokens(
    prompt: str,
    *,
    tokenizer: PreTrainedTokenizer,
    embedding_module,                 # e.g., text_model.get_input_embeddings()
    slot_embed: torch.Tensor,         # [D] or [num_slots, D]
    slot_token_literal: str = "*",
    # 물리 채널 설정
    channel_type: str = "awgn",       # 'awgn' | 'rayleigh'
    snr_db: float = 20.0,             # Eb/N0 (dB) — 디지털 기준
    fading: str = "vector",           # 'vector'|'element' — 아날로그 슬롯 임베딩에 적용
    # 디지털 전송(비-슬롯) 변조
    modulation: str = "qpsk",         # 'bpsk' | 'qpsk' | '16qam' | '64qam'
    freeze_special: bool = True,      # 특수토큰 보호
    # 수신 텍스트 렌더링
    render_placeholder_as: str = "*",
    rng: Optional[torch.Generator] = None,
) -> Dict[str, Union[str, torch.Tensor, torch.LongTensor, int]]:
    device = embedding_module.weight.device
    dtype  = embedding_module.weight.dtype
    V      = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    bits_per_token = int(math.ceil(math.log2(max(2, V))))

    # 0) 토크나이즈 및 마스크
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)      # [1,T]
    attention_mask = enc.attention_mask.to(device)
    B, T = input_ids.shape
    assert B == 1, "현재 단일 문장만 지원."

    toks = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    slot_mask = torch.tensor([_normalize_token_literal(t) == slot_token_literal for t in toks],
                             device=device, dtype=torch.bool).unsqueeze(0)  # [1,T]
    non_slot_mask = (~slot_mask)

    # 특수토큰 보호
    ids = input_ids.clone()
    if freeze_special and hasattr(tokenizer, "all_special_ids"):
        special = torch.tensor(tokenizer.all_special_ids, device=device)
        is_special = (ids.unsqueeze(-1) == special).any(dim=-1)  # [1,T]
    else:
        is_special = torch.zeros_like(ids, dtype=torch.bool)

    digi_mask = non_slot_mask & (~is_special) & (attention_mask.bool())

    # 1) 비-슬롯 디지털 경로
    ids = _digital_path_ids_through_qam(
        ids, digi_mask=digi_mask, vocab_size=V, bits_per_token=bits_per_token,
        channel_type=channel_type, snr_db=snr_db, modulation=modulation, rng=rng
    )
    recovered_input_ids = ids  # [1,T]

    # 2) 슬롯 아날로그 경로
    E = embedding_module
    base_embeds = E(recovered_input_ids).to(dtype)  # [1,T,D]
    if slot_mask.any():
        ph = slot_embed.to(device=device, dtype=dtype)
        num_slots = int(slot_mask.sum().item())
        if ph.ndim == 1:
            ph = ph.unsqueeze(0).expand(num_slots, -1)  # [n_occ,D]
        idxs = slot_mask[0].nonzero(as_tuple=False).flatten()
        slot_noisy = _apply_analog_on_vectors(
            ph, snr_db=snr_db, channel_type=channel_type, fading=fading, rng=rng
        )
        base_embeds[0, idxs, :] = slot_noisy

    inputs_embeds = base_embeds  # [1,T,D]

    # 3) 수신 텍스트 렌더링
    render_ids = recovered_input_ids.clone()
    if slot_mask.any():
        try:
            slot_id_for_render = tokenizer.convert_tokens_to_ids(slot_token_literal)
            if slot_id_for_render is None or slot_id_for_render < 0:
                raise ValueError
        except Exception:
            slot_id_for_render = getattr(tokenizer, "unk_token_id", int(render_ids[0, 0].item()))
        render_ids[0, slot_mask[0]] = slot_id_for_render

    recovered_text = tokenizer.batch_decode(render_ids, skip_special_tokens=True)[0]
    if render_placeholder_as != slot_token_literal and slot_mask.any():
        recovered_text = recovered_text.replace(slot_token_literal, render_placeholder_as)

    return {
        "recovered_text": recovered_text,
        "recovered_input_ids": recovered_input_ids,  # [1,T]
        "inputs_embeds": inputs_embeds,              # [1,T,D]
        "attention_mask": attention_mask,            # [1,T]
        "slot_mask": slot_mask,                      # [1,T]
        "bit_length_per_token": bits_per_token,
        "bits_per_symbol": _mod_params(modulation, device=device)[0],
        "assumption": "Digital path uses coherent detection with perfect CSI; Es normalized to 1; snr_db interpreted as Eb/N0 (dB).",
    }

@torch.no_grad()
def noisy_prompt_multi_analog_slots_discrete_tokens(
    prompt: str,
    *,
    tokenizer: PreTrainedTokenizer,
    embedding_module,                      # e.g., text_model.get_input_embeddings()
    slot_token_literals: List[str],        # 예: ["*", "<v1>", "<v2>"]
    slot_embed_map: Optional[Dict[str, torch.Tensor]] = None,
    # 물리 채널 설정
    channel_type: str = "awgn",            # 'awgn' | 'rayleigh'
    snr_db: float = 20.0,                  # Eb/N0 (dB)
    fading: str = "vector",                # 'vector'|'element'
    # 디지털 전송(비-슬롯) 변조
    modulation: str = "qpsk",              # 'bpsk'|'qpsk'|'16QAM'|'64QAM'
    freeze_special: bool = True,           # 특수토큰 보호
    # 수신 텍스트 렌더링
    render_placeholder_map: Optional[Dict[str, str]] = None,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, Union[str, torch.Tensor, torch.LongTensor, int, Dict[str, torch.Tensor], Dict[str, List[int]]]]:
    device = embedding_module.weight.device
    dtype  = embedding_module.weight.dtype
    V      = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    bits_per_token = int(math.ceil(math.log2(max(2, V))))

    # 0) 토크나이즈 및 마스크
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)      # [1,T]
    attention_mask = enc.attention_mask.to(device)
    B, T = input_ids.shape
    assert B == 1, "현재 단일 문장만 지원."

    toks = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    slot_masks: Dict[str, torch.Tensor] = {}
    for lit in slot_token_literals:
        slot_masks[lit] = torch.tensor(
            [_normalize_token_literal(t) == lit for t in toks],
            device=device, dtype=torch.bool
        ).unsqueeze(0)

    slot_mask_union = torch.zeros((1, T), device=device, dtype=torch.bool)
    for m in slot_masks.values():
        slot_mask_union |= m
    non_slot_mask = (~slot_mask_union)

    # 특수토큰 보호
    ids = input_ids.clone()
    if freeze_special and hasattr(tokenizer, "all_special_ids"):
        special = torch.tensor(tokenizer.all_special_ids, device=device)
        is_special = (ids.unsqueeze(-1) == special).any(dim=-1)
    else:
        is_special = torch.zeros_like(ids, dtype=torch.bool)

    digi_mask = non_slot_mask & (~is_special) & (attention_mask.bool())

    # 1) 비-슬롯 디지털 경로
    ids = _digital_path_ids_through_qam(
        ids, digi_mask=digi_mask, vocab_size=V, bits_per_token=bits_per_token,
        channel_type=channel_type, snr_db=snr_db, modulation=modulation, rng=rng
    )
    recovered_input_ids = ids  # [1,T]

    # 2) 슬롯(리터럴별) 아날로그 경로
    E = embedding_module
    base_embeds = E(recovered_input_ids).to(dtype)  # [1,T,D]
    D = base_embeds.shape[-1]

    noisy_slot_embed_map: Dict[str, torch.Tensor] = {}
    noisy_slot_embed_map_by_id: Dict[int, torch.Tensor] = {}
    per_occ_noisy: Dict[str, torch.Tensor] = {}
    slot_positions: Dict[str, List[int]] = {}

    def _auto_slot_embed(literal: str) -> torch.Tensor:
        tok_id = tokenizer.convert_tokens_to_ids(literal)
        if isinstance(tok_id, int) and tok_id >= 0:
            return E.weight[tok_id].detach().to(device=device, dtype=dtype)
        toks_local = tokenizer(literal, add_special_tokens=False, return_tensors="pt")
        ids_local = toks_local.input_ids.to(device)
        return E(ids_local).mean(dim=1).squeeze(0).detach().to(dtype)

    for lit, mask in slot_masks.items():
        idxs = mask[0].nonzero(as_tuple=False).flatten()
        slot_positions[lit] = idxs.tolist()
        if idxs.numel() == 0:
            continue

        # 주입 소스 선택
        src = None
        if slot_embed_map is not None and lit in slot_embed_map:
            src = slot_embed_map[lit].to(device=device, dtype=dtype)

        ph = src if src is not None else _auto_slot_embed(lit)
        if ph.ndim == 1:
            ph = ph.unsqueeze(0).expand(idxs.numel(), -1)  # [n_occ,D]
        elif ph.ndim == 2:
            if ph.size(0) == 1:
                ph = ph.expand(idxs.numel(), -1)
            elif ph.size(0) != idxs.numel():
                # 멀티벡터 블록 → 평균으로 per-occurrence 요약 주입
                ph = ph.mean(dim=0, keepdim=True).expand(idxs.numel(), -1)
        else:
            raise ValueError(f"slot_embed_map['{lit}'] must be [D] or [n_occ,D].")

        slot_noisy = _apply_analog_on_vectors(
            ph, snr_db=snr_db, channel_type=channel_type, fading=fading, rng=rng
        )
        base_embeds[0, idxs, :] = slot_noisy
        per_occ_noisy[lit] = slot_noisy

        lit_id = tokenizer.convert_tokens_to_ids(lit)
        if src is None:
            noisy_slot_embed_map[lit] = slot_noisy
            if isinstance(lit_id, int) and lit_id >= 0:
                noisy_slot_embed_map_by_id[lit_id] = slot_noisy
        else:
            if src.ndim == 2 and src.size(0) != idxs.numel():
                # [V,D] 블록 전체에도 채널 적용하여 별도로 저장
                block = src
                block_noisy = _apply_analog_on_vectors(
                    block, snr_db=snr_db, channel_type=channel_type, fading=fading, rng=rng
                )
                noisy_slot_embed_map[lit] = block_noisy
                if isinstance(lit_id, int) and lit_id >= 0:
                    noisy_slot_embed_map_by_id[lit_id] = block_noisy
            else:
                noisy_slot_embed_map[lit] = slot_noisy
                if isinstance(lit_id, int) and lit_id >= 0:
                    noisy_slot_embed_map_by_id[lit_id] = slot_noisy

    inputs_embeds = base_embeds  # [1,T,D]

    # 3) 수신 텍스트 렌더링
    render_ids = recovered_input_ids.clone()
    for lit, mask in slot_masks.items():
        try:
            lit_id = tokenizer.convert_tokens_to_ids(lit)
            if lit_id is None or lit_id < 0:
                raise ValueError
        except Exception:
            lit_id = getattr(tokenizer, "unk_token_id", int(render_ids[0, 0].item()))
        render_ids[0, mask[0]] = lit_id

    recovered_text = tokenizer.batch_decode(render_ids, skip_special_tokens=True)[0]
    if render_placeholder_map:
        for lit, rep in render_placeholder_map.items():
            recovered_text = recovered_text.replace(lit, rep)

    return {
        "recovered_text": recovered_text,
        "recovered_input_ids": recovered_input_ids,   # [1,T]
        "inputs_embeds": inputs_embeds,               # [1,T,D]
        "attention_mask": attention_mask,             # [1,T]
        "slot_mask": slot_mask_union,                 # [1,T]
        "slot_masks": slot_masks,                     # {literal: [1,T]}
        "slot_positions": slot_positions,             # {literal: [indices]}
        "noisy_slot_embed_map": noisy_slot_embed_map,
        "noisy_slot_embed_map_by_id": noisy_slot_embed_map_by_id,
        "bit_length_per_token": bits_per_token,
        "bits_per_symbol": _mod_params(modulation, device=device)[0],
        "assumption": (
            "Digital path uses coherent detection with perfect CSI; Es normalized to 1; "
            "snr_db interpreted as Eb/N0 (dB) for digital, same SNR base for analog slot path."
        ),
    }

def word_to_embed(word: str, tokenizer: CLIPTokenizer, embedding_module) -> torch.Tensor:
    with torch.no_grad():
        toks = tokenizer(word, add_special_tokens=False, return_tensors="pt")
        ids = toks.input_ids.to(embedding_module.weight.device)
        E = embedding_module
        emb = E(ids)  # [1, n_sub, D]
        return emb.mean(dim=1).squeeze(0)  # [D]


# ─────────────────────────────────────────────────────────────────────
#                                Demo
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import CLIPTextModel, CLIPTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    txt = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    emb = txt.get_input_embeddings()

    prompts = [
        "a photo of a * bear plushie on a wooden table",
        "a red sports car parked in front of a mountain"
    ]

    res = text_through_channel(
        prompts,
        tokenizer=tok,
        embedding_module=emb,
        channel_type="awgn",
        snr_db=-10.0,
        fading="vector",
        per_token=True,
        training=False,
        eval_add_noise=True,
        freeze_special=True,
        decode_strategy="nearest",
        return_inputs_embeds=True,
    )
    print(res["noisy_text"][0])
    print(res["noisy_text"][1])

    prompt = "a photo of * with blue guitar"
    slot_word = "bear"
    slot_embed = word_to_embed(slot_word, tok, emb)

    rng = torch.Generator(device=device).manual_seed(0)
    tests = [
        ("awgn", 5.0,  "bpsk"),
        ("awgn", 5.0,  "qpsk"),
        ("awgn", 5.0, "16qam"),
        ("awgn", 5.0, "64qam"),
        ("rayleigh", 5.0, "qpsk"),
        ("rayleigh", 5.0, "16qam"),
    ]

    print(f"\n[Prompt] {prompt}")
    print(f"[Slot word] {slot_word}  (slot_embed dim={slot_embed.numel()})\n")
    for ch, snr_db, mod in tests:
        res = noisy_prompt_analog_slot_discrete_tokens(
            prompt,
            tokenizer=tok,
            embedding_module=emb,
            slot_embed=slot_embed,
            slot_token_literal="*",
            channel_type=ch,
            snr_db=snr_db,
            fading="vector",
            modulation=mod,
            freeze_special=True,
            render_placeholder_as=slot_word,
            rng=rng,
        )
        print(f"=== Channel={ch.upper():8s}  Eb/N0={snr_db:>5.1f} dB  Mod={mod.upper():7s} ===")
        print("Recovered text:", res["recovered_text"])
        ids = res["recovered_input_ids"]
        print("IDs shape:", tuple(ids.shape), "| inputs_embeds shape:", tuple(res["inputs_embeds"].shape))
        print("bits/token:", res["bit_length_per_token"], "| bits/symbol:", res["bits_per_symbol"])
        print()
