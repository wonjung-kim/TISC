import clip
import os, time, re
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import tqdm as tq
import sklearn.preprocessing
from packaging import version
import warnings
from shutil import rmtree, copy
from diffusers import DiffusionPipeline
from transformers import ViTModel
from dreamsim import dreamsim
from lpips import LPIPS
import json, shutil
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from channel import noisy_prompt_multi_analog_slots_discrete_tokens
from util_functions import log_validation as _run_log_validation

def _load_for_lpips(paths, device, img_size=256):
    tfm = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    batch = []
    for p in paths:
        with Image.open(p) as im:
            img = Image.open(p).convert("RGB")
        t = tfm(img).unsqueeze(0)
        batch.append(t)
    if not batch:
        return torch.empty(0,3,img_size,img_size, device=device)
    return torch.cat(batch, dim=0).to(device)

@torch.inference_mode()
def lpips_best_per_gen(gen_paths, ref_paths, lpips_model, device="cuda", batch_size=16):
    if not gen_paths or not ref_paths:
        return np.array([])  # empty
    G = _load_for_lpips(gen_paths, device)  # (Ng,3,H,W)
    R = _load_for_lpips(ref_paths, device)  # (Nr,3,H,W)
    Ng, Nr = len(G), len(R)
    best_dist = torch.full((Ng,), float("inf"), device=device)
    for gi in range(0, Ng, batch_size):
        g = G[gi:gi+batch_size]                           # (gb,3,H,W)
        cur_best = torch.full((g.size(0),), float("inf"), device=device)
        for ri in range(0, Nr, batch_size):
            r = R[ri:ri+batch_size]                       # (rb,3,H,W)
            # cartesian pairs
            g_flat = g.unsqueeze(1).expand(-1, r.size(0), -1, -1, -1).reshape(-1, *g.shape[1:])
            r_flat = r.unsqueeze(0).expand(g.size(0), -1, -1, -1, -1).reshape(-1, *r.shape[1:])
            d = lpips_model(g_flat, r_flat).view(g.size(0), r.size(0))  # (gb,rb)
            cur_best = torch.minimum(cur_best, d.min(dim=1).values)
        best_dist[gi:gi+batch_size] = torch.minimum(best_dist[gi:gi+batch_size], cur_best)
    return best_dist.detach().cpu().numpy()  # shape (Ng,)

# ----------------------- DreamSim 전용 헬퍼 -------------------
def _load_imgs_for_dreamsim(paths, preprocess, device):
    """List[str] → (N,C,H,W) tensor  (개별 이미지 전처리)"""
    tensors = []
    for p in paths:
        with Image.open(p) as im:
            img = Image.open(p).convert("RGB")
        t = preprocess(img)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        tensors.append(t)
    if not tensors:
        return torch.empty(0, 3, 224, 224, device=device)
    return torch.cat(tensors, dim=0).to(device)

@torch.inference_mode()
def dreamsim_best_per_gen(gen_paths, ref_paths, model, preprocess, device="cuda", batch_size=16):
    if not gen_paths or not ref_paths:
        return np.array([])  # empty
    G = _load_imgs_for_dreamsim(gen_paths, preprocess, device)  # (Ng,3,224,224)
    R = _load_imgs_for_dreamsim(ref_paths, preprocess, device)  # (Nr,3,224,224)
    Ng, Nr = len(G), len(R)
    best_dist = torch.full((Ng,), float("inf"), device=device)
    for gi in range(0, Ng, batch_size):
        g_chunk = G[gi:gi+batch_size]                      # (gb,3,224,224)
        cur_best = torch.full((g_chunk.size(0),), float("inf"), device=device)
        for ri in range(0, Nr, batch_size):
            r_chunk = R[ri:ri+batch_size]                  # (rb,3,224,224)
            pairs = g_chunk.size(0) * r_chunk.size(0)
            g_flat = g_chunk.unsqueeze(1).expand(-1, r_chunk.size(0), -1, -1, -1
                      ).contiguous().view(pairs, 3, 224, 224)
            r_flat = r_chunk.unsqueeze(0).expand(g_chunk.size(0), -1, -1, -1, -1
                      ).contiguous().view(pairs, 3, 224, 224)
            d = model(g_flat, r_flat).view(g_chunk.size(0), r_chunk.size(0))  # (gb,rb)
            cur_best = torch.minimum(cur_best, d.min(dim=1).values)
        best_dist[gi:gi+batch_size] = torch.minimum(best_dist[gi:gi+batch_size], cur_best)
    best_sim = (1.0 - best_dist).clamp(min=0.0, max=1.0).detach().cpu().numpy()
    return best_sim  # shape (Ng,)

@torch.inference_mode()
def dreamsim_i2i_score(gen_paths, ref_paths, model, preprocess,
                       device="cuda", batch_size=16):
    """
    두 이미지 집합(gen, ref) 간 DreamSim 평균 *유사도* 반환.
    DreamSim은 distance(0=identical, ↑=다름)이므로
    여기서는  similarity = 1 - distance  로 변환 (0~1 범위 권장).
    """
    g = _load_imgs_for_dreamsim(gen_paths, preprocess, device)
    r = _load_imgs_for_dreamsim(ref_paths, preprocess, device)

    if len(g) == 0 or len(r) == 0:
        return 0.0

    total, cnt = 0.0, 0
    for gi in range(0, len(g), batch_size):
        g_chunk = g[gi: gi + batch_size]                # (g_b,C,H,W)
        for ri in range(0, len(r), batch_size):
            r_chunk = r[ri: ri + batch_size]            # (r_b,C,H,W)
            # Cartesian product -> 한 번에 모델 호출
            pairs = g_chunk.size(0) * r_chunk.size(0)
            g_flat = g_chunk.unsqueeze(1).expand(-1, r_chunk.size(0), -1, -1, -1
                        ).contiguous().view(pairs, 3, 224, 224)
            r_flat = r_chunk.unsqueeze(0).expand(g_chunk.size(0), -1, -1, -1, -1
                        ).contiguous().view(pairs, 3, 224, 224)

            d = model(g_flat, r_flat)                  # (pairs,) distances
            total += d.sum().item()
            cnt   += d.numel()

    avg_dist = total / cnt
    return 1.0 - avg_dist                              # similarity

class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return transforms.Compose([
            transforms.Resize(n_px, interpolation=Image.BICUBIC),
            transforms.CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)

class DINOImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        """
        images: PIL.Image.Image 객체 리스트 혹은 (추후) 파일 경로 리스트
        """
        self.images = images

        # 전처리를 __init__에서 정의
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # 만약 파일 경로(str)가 넘어올 가능성이 있으면, 여기서 PIL로 로드
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        # 미리 정의해 둔 self.transform을 적용
        img_t = self.transform(img)
        return {"image": img_t}
    

def extract_all_dino_images(images, dino_model, device, batch_size=64, num_workers=2):
    """
    ViTModel (DINO) 를 이용해 PIL.Image 리스트에서 CLS 토큰 임베딩 (NumPy) 반환
    """
    dataset = DINOImageDataset(images)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    all_feats = []
    dino_model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)            # (B, 3, 224, 224)
            outputs = dino_model(x)                 # ViTModelOutput
            cls_feats = outputs.last_hidden_state[:, 0]  # (B, D)
            all_feats.append(cls_feats.cpu().numpy())

    all_feats = np.vstack(all_feats)  # (N, D)
    return all_feats


def extract_all_clip_images(images, model, device, batch_size=64, num_workers=2):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tq.tqdm(data):
            b = b['image'].to(device)
            b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def extract_all_captions(captions, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tq.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def get_similarity_score(model, clip_images, original_images, device, extract_fn, w=1.0):
    if isinstance(clip_images, list):
        # need to extract image features
        clip_images = extract_fn(clip_images, model, device)
    if isinstance(original_images, list):
        # need to extract image features
        original_images = extract_fn(original_images, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        clip_images = sklearn.preprocessing.normalize(clip_images, axis=1)
        original_images = sklearn.preprocessing.normalize(original_images, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than'
            'paper results. To exactly replicate paper results, please use numpy version less'
            'than 1.21, e.g., 1.20.3.')
        clip_images = clip_images / np.sqrt(np.sum(clip_images ** 2, axis=1, keepdims=True))
        original_images = original_images / np.sqrt(np.sum(original_images ** 2, axis=1,
                                                           keepdims=True))

    per = w * np.clip(np.dot(clip_images, original_images.T), 0, None)
    return np.mean(per)


# ─────────────────────────────────────────────────────────────────────────────
#  텍스트 인코더 통과용 유틸
# ─────────────────────────────────────────────────────────────────────────────
def encode_text(prompt: str, tokenizer, text_encoder, device):
    tok = tokenizer(
        [prompt], padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = text_encoder(input_ids=tok.input_ids,
                           attention_mask=tok.attention_mask, return_dict=True)
    return out.last_hidden_state, tok.attention_mask  # [1,S,H], [1,S]


# ─────────────────────────────────────────────────────────────────────────────
#  멀티벡터 임베딩을 placeholder 시작 위치부터 연속 슬롯에 주입
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def make_prompt_embeds_with_multivector_multi(
    prompt: str,
    new_embed_map: dict,    # {placeholder_id_or_literal: [V,H]}
    tokenizer,
    text_encoder,
    device,
    return_debug: bool = False,
):
    tok = tokenizer([prompt], padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
    input_ids = tok.input_ids            # [1,S]
    attn_mask = tok.attention_mask       # [1,S]

    # 0) 키 해석기: str/텐서/넘파이 → int 토큰 ID
    def _resolve_pid(pid_key) -> int:
        # 이미 int
        if isinstance(pid_key, int):
            return pid_key
        # 0-dim torch.Tensor
        if isinstance(pid_key, torch.Tensor):
            if pid_key.numel() == 1:
                return int(pid_key.item())
            raise TypeError(f"placeholder key tensor must be scalar, got shape {tuple(pid_key.shape)}")
        # numpy 정수
        try:
            import numpy as np
            if isinstance(pid_key, (np.integer,)):
                return int(pid_key)
        except Exception:
            pass
        # 문자열 리터럴 → 토큰 ID 해석
        if isinstance(pid_key, str):
            tid = tokenizer.convert_tokens_to_ids(pid_key)
            if isinstance(tid, int) and tid >= 0:
                return tid
            # 단어가 다중 서브워드로 쪼개지는 경우는 현재 지원 X (한 토큰 전제)
            ids_local = tokenizer(pid_key, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
            if len(ids_local) == 1:
                return int(ids_local[0])
            raise ValueError(
                f"Placeholder '{pid_key}' maps to {len(ids_local)} subword tokens {ids_local}. "
                f"Please pass a single-token literal or its token id."
            )
        raise TypeError(f"Unsupported placeholder key type: {type(pid_key)}")

    # 1) token embedding 뽑기
    E = text_encoder.get_input_embeddings()
    inputs_embeds = E(input_ids)         # [1,S,H]
    S, H = inputs_embeds.size(1), inputs_embeds.size(2)

    # 2) placeholder별로 '모든 위치'에 블록 주입
    debug_info = {}
    for pid_key, block in new_embed_map.items():
        pid = _resolve_pid(pid_key)  # 🔴 키를 반드시 int 토큰 ID로 변환
        if not isinstance(block, torch.Tensor):
            raise TypeError(f"block for id={pid_key} must be torch.Tensor, got {type(block)}")
        assert block.dim() == 2 and block.size(1) == H, f"block for id={pid_key} must be [V,{H}]"
        V = block.size(0)

        # 텐서 비교로 위치 탐색 (항상 bool 텐서가 되도록 보장)
        mask = (input_ids[0] == int(pid))           # [S] bool tensor
        pos = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        debug_info[int(pid)] = {"positions": pos, "V": int(V)}

        for start in pos:
            end = min(start + V, S)
            k = end - start
            if k <= 0:
                continue
            if k < V:
                print(f"[WARN] Placeholder id={pid} at pos={start}: sequence limit -> inject only {k}/{V}")
            inputs_embeds[0, start:start+k, :] = block[:k, :].to(inputs_embeds.dtype, non_blocking=True)

    # 3) position embedding + encoder + LN
    pos_ids = text_encoder.text_model.embeddings.position_ids[:, :S]
    pos_embeds = text_encoder.text_model.embeddings.position_embedding(pos_ids)
    hidden = inputs_embeds + pos_embeds

    causal_mask = _create_4d_causal_attention_mask((1, S), hidden.dtype, hidden.device)
    enc_out = text_encoder.text_model.encoder(
        inputs_embeds=hidden,
        attention_mask=None,
        causal_attention_mask=causal_mask,
        return_dict=True
    )
    last_hidden = text_encoder.text_model.final_layer_norm(enc_out.last_hidden_state)
    return (last_hidden, attn_mask, debug_info) if return_debug else (last_hidden, attn_mask)


def score_computation(
    text_encoder, tokenizer, unet, vae, args,
    weight_dtype, logger,
    val_prompt,
    base_placeholder_ids: list[int],
    learned_weights: torch.Tensor,
    base_placeholder_literals: list[str],
):
    """
    Validation과 동일한 환경에서 이미지를 생성한 뒤,
    metric 분포를 CSV로 저장(Seaborn error bars 대비)하고
    기존 best 결과도 함께 반환합니다.
    """
    import shutil, json, csv, math, random
    import numpy as np
    import torch
    from pathlib import Path
    from diffusers import DiffusionPipeline
    from tqdm.auto import tqdm  # ✅ 진행바

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 플래그/하이퍼파라미터 (검증과 동기화) ----------
    channel_type = getattr(args, "text_channel_type", getattr(args, "channel_type", "none"))
    snr_db = float(getattr(args, "text_snr_db", getattr(args, "snr_db", 20.0)))
    text_eval_add_noise = bool(getattr(args, "text_eval_add_noise", True))
    freeze_special = bool(getattr(args, "text_freeze_special", True))
    keep_placeholder_clean = bool(getattr(args, "text_keep_placeholder_clean", True))

    num_inference_steps = int(getattr(args, "validation_num_inference_steps",
                               getattr(args, "num_inference_steps", 50)))
    guidance_scale = float(getattr(args, "validation_guidance_scale",
                            getattr(args, "guidance_scale", 7.5)))

    target_ber = float(getattr(args, "text_target_ber", 1e-2))
    text_modulation = getattr(args, "text_modulation", "qpsk")
    cat_tag = getattr(args, "category", "unknown")

    def _pick_mod_by_snr(snr_db: float, target_ber: float = 1e-2, rayleigh: bool = False) -> str:
        margin = 2.0 if rayleigh else 0.0
        thr_bpsk_qpsk = 7.0 + margin
        thr_qpsk_16   = 12.0 + margin
        thr_16_64     = 18.0 + margin
        if snr_db < thr_bpsk_qpsk:  return "bpsk"
        if snr_db < thr_qpsk_16:    return "qpsk"
        if snr_db < thr_16_64:      return "16qam"
        return "64qam"
    if str(text_modulation).lower() == "adaptive":
        text_mod = _pick_mod_by_snr(snr_db, target_ber=target_ber, rayleigh=(channel_type == "rayleigh"))
    else:
        text_mod = text_modulation

    bootstrap_n = int(getattr(args, "metrics_bootstrap_samples", 0))
    conf_level = float(getattr(args, "metrics_confidence", 0.95))

    # ---------- 파이프라인 ----------
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        revision=getattr(args, "revision", None),
        torch_dtype=weight_dtype,
        safety_checker=None,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # ---------- 멀티벡터 준비 ----------
    P = len(base_placeholder_ids)
    V = int(getattr(args, "num_vectors_per_token", 1))
    assert learned_weights.shape[0] == P * V, "learned_weights.shape[0] must be P*V"

    new_embed_map = {}
    for p_idx, base_id in enumerate(base_placeholder_ids):
        block = learned_weights[p_idx*V:(p_idx+1)*V, :].to(device)
        new_embed_map[base_id] = block

    from channel import noisy_prompt_multi_analog_slots_discrete_tokens

    def _make_prompt_embeds(prompt_text: str, embed_map):
        # 동일 모듈 내에 정의되어 있다고 가정
        pe, _ = make_prompt_embeds_with_multivector_multi(
            prompt_text, embed_map, tokenizer, text_encoder, device
        )
        return pe

    # ---------- 생성 ----------
    N = int(getattr(args, "score_number", 64))
    out_root = Path(args.output_dir)
    gen_dir = out_root / "generated_for_metrics"
    best_dir = out_root / "best_by_metric"
    gen_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    base_prompt = val_prompt
    seed0 = getattr(args, "seed", None)

    if (not text_eval_add_noise) or (str(channel_type).lower() == "none"):
        desc = f"[{cat_tag}] Generating (clean) | steps={num_inference_steps}, scale={guidance_scale}"
        pe = _make_prompt_embeds(base_prompt, new_embed_map)
        for n in tqdm(range(N), desc=desc, unit="img"):
            gen = torch.Generator(device=device).manual_seed(seed0 + n) if seed0 is not None else None
            img = pipeline(prompt_embeds=pe,
                           num_inference_steps=num_inference_steps,
                           guidance_scale=guidance_scale,
                           generator=gen).images[0]
            img.save(gen_dir / f"gen_{n+1:03d}.png")
    else:
        desc = f"[{cat_tag}] Generating ({channel_type}, {snr_db} dB, mod={text_mod})"
        for n in tqdm(range(N), desc=desc, unit="img"):
            gen = torch.Generator(device=device).manual_seed(seed0 + n) if seed0 is not None else None
            noisy = noisy_prompt_multi_analog_slots_discrete_tokens(
                prompt=base_prompt,
                tokenizer=tokenizer,
                embedding_module=text_encoder.get_input_embeddings(),
                slot_token_literals=base_placeholder_literals,
                slot_embed_map=new_embed_map,
                channel_type=channel_type,
                snr_db=snr_db,
                fading="vector",
                modulation=text_mod,
                freeze_special=freeze_special,
                render_placeholder_map=None,
                rng=gen,
            )
            noisy_prompt = noisy["recovered_text"]
            if keep_placeholder_clean:
                for lit in base_placeholder_literals:
                    if lit not in noisy_prompt:
                        noisy_prompt = base_prompt
                        noisy["noisy_slot_embed_map"] = new_embed_map
                        break
            noisy_embed_map = noisy["noisy_slot_embed_map"]
            pe = _make_prompt_embeds(noisy_prompt, noisy_embed_map)
            img = pipeline(prompt_embeds=pe,
                           num_inference_steps=num_inference_steps,
                           guidance_scale=guidance_scale,
                           generator=gen).images[0]
            img.save(gen_dir / f"gen_{n+1:03d}.png")

    # ---------- 레퍼런스 ----------
    td = Path(args.train_data_dir)
    # 기본값: 00.jpg만 사용 (필요하면 args.ref_filename으로 덮어쓰기 지원)
    ref_name = getattr(args, "ref_filename", "00.jpg")
    ref_file = td / ref_name

    if ref_file.exists():
        ref_paths = [str(ref_file)]
    else:
        logger.warning(f"Reference image '{ref_name}' not found in {td}. Aborting metrics for this run.")
        return {}
    gen_paths = sorted([str(p) for p in gen_dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.tiff'}])
    if not ref_paths or not gen_paths:
        logger.warning("No reference or generated images for metrics.")
        return {}

    # ---------- Metric 모델/추출 ----------
    # 여기서는 외부 추출 함수들을 쓰므로 coarse 단계 진행바로 표시
    with tqdm(total=4, desc="Computing metrics", unit="stage") as pbar:
        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False); clip_model.eval()
        clip_gen = extract_all_clip_images(gen_paths, clip_model, device, batch_size=N, num_workers=2)
        clip_ref = extract_all_clip_images(ref_paths, clip_model, device, batch_size=len(ref_paths), num_workers=2)
        clip_gen = clip_gen / np.clip(np.linalg.norm(clip_gen, axis=1, keepdims=True), 1e-8, None)
        clip_ref = clip_ref / np.clip(np.linalg.norm(clip_ref, axis=1, keepdims=True), 1e-8, None)
        sims = clip_gen @ clip_ref.T
        clip_best_per_gen = sims.max(axis=1)
        clip_best_idx = int(np.argmax(clip_best_per_gen))
        clip_best_score = float(clip_best_per_gen[clip_best_idx])
        pbar.update(1)

        dino_model = ViTModel.from_pretrained("facebook/dino-vits16").to(device).eval()
        dino_gen = extract_all_dino_images(gen_paths, dino_model, device, batch_size=N, num_workers=2)
        dino_ref = extract_all_dino_images(ref_paths, dino_model, device, batch_size=len(ref_paths), num_workers=2)
        dino_gen = dino_gen / np.clip(np.linalg.norm(dino_gen, axis=1, keepdims=True), 1e-8, None)
        dino_ref = dino_ref / np.clip(np.linalg.norm(dino_ref, axis=1, keepdims=True), 1e-8, None)
        dino_sims = dino_gen @ dino_ref.T
        dino_best_per_gen = dino_sims.max(axis=1)
        dino_best_idx = int(np.argmax(dino_best_per_gen))
        dino_best_score = float(dino_best_per_gen[dino_best_idx])
        pbar.update(1)

        dream_model, dream_preproc = dreamsim(pretrained=True, device=device)
        dream_best_per_gen = dreamsim_best_per_gen(gen_paths, ref_paths, dream_model, dream_preproc,
                                                   device=device, batch_size=16)
        dream_best_idx = int(np.argmax(dream_best_per_gen))
        dream_best_score = float(dream_best_per_gen[dream_best_idx])
        pbar.update(1)

        lpips_model = LPIPS(net="vgg").to(device).eval()
        lpips_best_dists = lpips_best_per_gen(gen_paths, ref_paths, lpips_model, device=device, batch_size=16)
        lpips_best_idx = int(np.argmin(lpips_best_dists))
        lpips_best_dist = float(lpips_best_dists[lpips_best_idx])
        lpips_best_sim = float(1.0 / (1.0 + lpips_best_dist))
        pbar.update(1)

    # ---------- Best 이미지 저장 ----------
    clip_best_path  = str(best_dir / "best_clip.png")
    dino_best_path  = str(best_dir / "best_dino.png")
    dream_best_path = str(best_dir / "best_dreamsim.png")
    lpips_best_path = str(best_dir / "best_lpips.png")
    shutil.copyfile(gen_paths[clip_best_idx],  clip_best_path)
    shutil.copyfile(gen_paths[dino_best_idx],  dino_best_path)
    shutil.copyfile(gen_paths[dream_best_idx], dream_best_path)
    shutil.copyfile(gen_paths[lpips_best_idx], lpips_best_path)

    # ---------- CSV 저장(Seaborn용 롱포맷/요약) ----------
    meta = {
        "snr_db": snr_db,
        "channel_type": channel_type,
        "modulation": text_mod,
        "eval_text_noise": bool(text_eval_add_noise and str(channel_type).lower() != "none"),
        "seed": getattr(args, "seed", None),
        "output_dir": str(args.output_dir),
        "category": getattr(args, "json_category", getattr(args, "category_name", None)),
        "jindex": getattr(args, "json_index", None),
    }
    per_image_rows = []
    for idx, val in enumerate(tqdm(clip_best_per_gen.tolist(), desc=f"[{cat_tag}] Aggregate per-image: CLIP", unit="img")):
        per_image_rows.append({"metric":"clip","value":float(val),"image_idx":idx, **meta})
    for idx, val in enumerate(tqdm(dino_best_per_gen.tolist(), desc=f"[{cat_tag}] Aggregate per-image: DINO", unit="img")):
        per_image_rows.append({"metric":"dino","value":float(val),"image_idx":idx, **meta})
    for idx, val in enumerate(tqdm(dream_best_per_gen.tolist(), desc=f"[{cat_tag}] Aggregate per-image: DreamSim", unit="img")):
        per_image_rows.append({"metric":"dreamsim","value":float(val),"image_idx":idx, **meta})
    for idx, val in enumerate(tqdm(lpips_best_dists.tolist(), desc=f"[{cat_tag}] Aggregate per-image: LPIPS", unit="img")):
        per_image_rows.append({"metric":"lpips","value":float(val),"image_idx":idx, **meta})

    per_image_csv = best_dir / "metrics_per_image.csv"
    write_header = not per_image_csv.exists()
    with per_image_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(per_image_rows)

    def _summary(arr):
        a = np.asarray(arr, dtype=float)
        n = a.size
        mean = float(a.mean()) if n else float("nan")
        std = float(a.std(ddof=1)) if n > 1 else float("nan")
        se = float(std / math.sqrt(n)) if n > 1 else float("nan")
        ci_low = ci_high = float("nan")
        if bootstrap_n and n > 1:
            means = []
            rng = random.Random(0xC0FFEE)
            for _ in tqdm(range(bootstrap_n), desc=f"[{cat_tag}] Bootstrap CIs", unit="samp"):
                sample = [a[rng.randrange(0, n)] for __ in range(n)]
                means.append(float(np.mean(sample)))
            means.sort()
            lo_idx = int((1.0 - conf_level)/2.0 * bootstrap_n)
            hi_idx = int((1.0 + conf_level)/2.0 * bootstrap_n) - 1
            lo_idx = max(0, min(bootstrap_n - 1, lo_idx))
            hi_idx = max(0, min(bootstrap_n - 1, hi_idx))
            ci_low = means[lo_idx]; ci_high = means[hi_idx]
        return {"n": n, "mean": mean, "std": std, "se": se, "ci_low": ci_low, "ci_high": ci_high}

    summary = {
        "clip":     _summary(clip_best_per_gen),
        "dino":     _summary(dino_best_per_gen),
        "dreamsim": _summary(dream_best_per_gen),
        "lpips":    _summary(lpips_best_dists),
    }
    for m in summary.values():
        m.update(meta)

    summary_csv = best_dir / "metrics_summary.csv"
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["metric","n","mean","std","se","ci_low","ci_high",
                      "snr_db","channel_type","modulation","eval_text_noise",
                      "seed","output_dir","category","jindex"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for name, stats in summary.items():
            row = {"metric": name, **{k: stats.get(k) for k in fieldnames if k != "metric"}}
            w.writerow(row)

    results = {
        "clip":      {"best_score": clip_best_score,   "best_image": clip_best_path},
        "dino":      {"best_score": dino_best_score,   "best_image": dino_best_path},
        "dreamsim":  {"best_score": dream_best_score,  "best_image": dream_best_path},
        "lpips":     {"best_distance": lpips_best_dist,
                      "best_similarity": lpips_best_sim,
                      "best_image": lpips_best_path},
        "mean": {
            "clip": float(sims.mean()),
            "dino": float(dino_sims.mean()),
            "dreamsim": float(dream_best_per_gen.mean()),
            "lpips": float(lpips_best_dists.mean()),
        },
        "distributions": {
            "clip_best_per_gen": clip_best_per_gen.tolist(),
            "dino_best_per_gen": dino_best_per_gen.tolist(),
            "dream_best_per_gen": dream_best_per_gen.tolist(),
            "lpips_best_dists": lpips_best_dists.tolist(),
            "per_image_csv": str(per_image_csv),
        },
        "summary": {
            "stats": summary,
            "summary_csv": str(summary_csv),
            "meta": meta,
        },
        "generation": {
            "channel_type": channel_type,
            "snr_db": snr_db,
            "text_modulation": text_mod,
            "text_eval_add_noise": text_eval_add_noise,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
    }
    with open(best_dir / "metrics_best.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    tqdm.write(f"[{cat_tag}] [SNR {snr_db}dB | {channel_type} | mod={text_mod}] "
               f"BEST - CLIP:{clip_best_score:.4f}  DINO:{dino_best_score:.4f}  "
               f"DreamSim:{dream_best_score:.4f}  LPIPS↓:{lpips_best_dist:.4f}")

    del pipeline
    return results


def only_score_without_generation(
    args, logger
):
    """
    - 각 metric별 최고 점수 이미지와 값 저장
      * CLIP / DINO / DreamSim: max(similarity)
      * LPIPS: min(distance)
    - 반환: metrics dict (best 점수/이미지 경로 포함)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Metric 모델 ----------
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False); clip_model.eval()
    dino_model = ViTModel.from_pretrained("facebook/dino-vits16").to(device).eval()
    dream_model, dream_preproc = dreamsim(pretrained=True, device=device)
    lpips_model = LPIPS(net="vgg").to(device).eval()

    gen_dir = Path(args.output_dir) / "generated_for_metrics"
    best_dir = Path(args.output_dir) / "best_by_metric"
    gen_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 레퍼런스 ----------
    td = Path(args.train_data_dir)
    # 딱 1장: 00.jpg만 사용
    ref_file = td / "00.jpg"
    if ref_file.exists():
        ref_paths = [str(ref_file)]
    else:
        logger.warning(f'"00.jpg" not found in {td}.')
        return {}

    gen_paths = sorted([str(p) for p in gen_dir.iterdir()
                        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff'}])

    if not ref_paths or not gen_paths:
        logger.warning("No reference or generated images for metrics.")
        return {}

    # ---------- CLIP per-gen best ----------
    clip_gen = extract_all_clip_images(gen_paths, clip_model, device, batch_size=32, num_workers=2)
    clip_ref = extract_all_clip_images(ref_paths, clip_model, device, batch_size=len(ref_paths), num_workers=2)
    # L2 normalize
    clip_gen = clip_gen / np.clip(np.linalg.norm(clip_gen, axis=1, keepdims=True), 1e-8, None)
    clip_ref = clip_ref / np.clip(np.linalg.norm(clip_ref, axis=1, keepdims=True), 1e-8, None)
    sims = clip_gen @ clip_ref.T                              # (Ng, Nr)
    clip_best_per_gen = sims.max(axis=1)                      # (Ng,)
    clip_best_idx = int(np.argmax(clip_best_per_gen))
    clip_best_score = float(clip_best_per_gen[clip_best_idx])

    # ---------- DINO per-gen best ----------
    dino_gen = extract_all_dino_images(gen_paths, dino_model, device, batch_size=32, num_workers=2)
    dino_ref = extract_all_dino_images(ref_paths, dino_model, device, batch_size=len(ref_paths), num_workers=2)
    # L2 normalize
    dino_gen = dino_gen / np.clip(np.linalg.norm(dino_gen, axis=1, keepdims=True), 1e-8, None)
    dino_ref = dino_ref / np.clip(np.linalg.norm(dino_ref, axis=1, keepdims=True), 1e-8, None)
    dino_sims = dino_gen @ dino_ref.T                         # (Ng, Nr)
    dino_best_per_gen = dino_sims.max(axis=1)
    dino_best_idx = int(np.argmax(dino_best_per_gen))
    dino_best_score = float(dino_best_per_gen[dino_best_idx])

    # ---------- DreamSim per-gen best ----------
    dream_best_per_gen = dreamsim_best_per_gen(gen_paths, ref_paths, dream_model, dream_preproc,
                                               device=device, batch_size=16)  # similarity
    dream_best_idx = int(np.argmax(dream_best_per_gen))
    dream_best_score = float(dream_best_per_gen[dream_best_idx])

    # ---------- LPIPS per-gen best (min distance) ----------
    lpips_best_dists = lpips_best_per_gen(gen_paths, ref_paths, lpips_model, device=device, batch_size=16)
    lpips_best_idx = int(np.argmin(lpips_best_dists))
    lpips_best_dist = float(lpips_best_dists[lpips_best_idx])
    lpips_best_sim = float(1.0 / (1.0 + lpips_best_dist))

    # ---------- best 이미지 저장 ----------
    clip_best_path  = str(best_dir / "best_clip.png")
    dino_best_path  = str(best_dir / "best_dino.png")
    dream_best_path = str(best_dir / "best_dreamsim.png")
    lpips_best_path = str(best_dir / "best_lpips.png")
    shutil.copyfile(gen_paths[clip_best_idx],  clip_best_path)
    shutil.copyfile(gen_paths[dino_best_idx],  dino_best_path)
    shutil.copyfile(gen_paths[dream_best_idx], dream_best_path)
    shutil.copyfile(gen_paths[lpips_best_idx], lpips_best_path)

    # ---------- 결과 저장(JSON) ----------
    results = {
        "clip":      {"best_score": clip_best_score,   "best_image": clip_best_path},
        "dino":      {"best_score": dino_best_score,   "best_image": dino_best_path},
        "dreamsim":  {"best_score": dream_best_score,  "best_image": dream_best_path},
        "lpips":     {"best_distance": lpips_best_dist,
                      "best_similarity": lpips_best_sim,
                      "best_image": lpips_best_path},
        # 평균값
        "mean": {
            "clip": float(sims.mean()),
            "dino": float(dino_sims.mean()),
            "dreamsim": float(dream_best_per_gen.mean()),
            "lpips": float(lpips_best_dists.mean()),
        }
    }
    with open(best_dir / "metrics_best.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"[SNR {args.snr_db}dB | {args.channel_type}] "
        f"BEST - CLIP:{clip_best_score:.4f}  DINO:{dino_best_score:.4f}  "
        f"DreamSim:{dream_best_score:.4f}  LPIPS↓:{lpips_best_dist:.4f}"
    )

    return results


def score_computation_v2(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    weight_dtype,
    logger,
    val_prompt: str,
    base_placeholder_ids: list[int],
    learned_weights: torch.Tensor,         # [P*V, H]
    base_placeholder_literals: list[str],
    accelerator,                           # ★ 추가: Accelerator 인스턴스
):
    """
    Accelerator 환경에서:
      1) 현재 args.snr_db(or args.text_snr_db)의 SNR로 args.score_number 장 생성 (util_functions.log_validation)
      2) 방금 생성한 묶음만 메트릭 계산 (CLIP/DINO/DreamSim/LPIPS)
      3) 결과/베스트 이미지 저장

    - 플레이스홀더(여러 개/P개, 각 V 벡터) 학습 가중치 learned_weights는
      '토큰 임베딩 테이블'에 임시 주입(평균) 후 생성 → 끝나면 원복.
    - Accelerator를 통해 device/unwrap_model 사용.
    """
    device = accelerator.device

    out_root = Path(args.output_dir)
    best_dir = out_root / "best_by_metric"
    best_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- args 패치 (생성용) ----------------
    # score_number 만큼 생성하도록
    orig_num_validation_images = int(getattr(args, "num_validation_images", 0) or 0)
    orig_text_snr_db = float(getattr(args, "text_snr_db", getattr(args, "snr_db", 20.0)))
    orig_text_channel_type = getattr(args, "text_channel_type", None)
    orig_placeholder_token = getattr(args, "placeholder_token", None)

    # log_validation은 text_* 키를 사용하므로 동기화
    cur_snr = float(getattr(args, "snr_db", getattr(args, "text_snr_db", 20.0)))
    setattr(args, "num_validation_images", int(getattr(args, "score_number", 64)))
    setattr(args, "text_snr_db", cur_snr)
    setattr(args, "text_channel_type", getattr(args, "channel_type", "none"))
    if base_placeholder_literals:
        setattr(args, "placeholder_token", base_placeholder_literals[0])  # 노이즈 타깃 토큰(필요 시)

    val_dir  = out_root / "generated_images"
    # ---------------- learned_W → 임베딩 테이블 임시 주입 ----------------
    # learned_weights: [P*V, H]  /  P=len(base_placeholder_ids)
    P = max(1, len(base_placeholder_ids))
    if learned_weights.ndim != 2 or learned_weights.shape[0] % P != 0:
        raise ValueError(f"learned_weights must be [P*V, H]; got {tuple(learned_weights.shape)} with P={P}")
    V = learned_weights.shape[0] // P
    H = learned_weights.shape[1]

    te_raw = accelerator.unwrap_model(text_encoder)
    emb = te_raw.get_input_embeddings()
    W  = emb.weight.data  # [Vocab, H]
    if W.shape[1] != H:
        raise ValueError(f"Embedding width mismatch: text_encoder.H={W.shape[1]} vs learned.H={H}")

    # 백업 후 주입(평균)
    backup = {}
    with torch.no_grad():
        for g, tid in enumerate(base_placeholder_ids):
            tid = int(tid)
            if not (0 <= tid < W.size(0)):
                logger.warning(f"[score] invalid placeholder id {tid} skipped.")
                continue
            block = learned_weights[g*V:(g+1)*V, :]  # [V,H]
            vec   = block.mean(dim=0, keepdim=False) # [H]  (원하면 block[0] 사용 가능)
            backup[tid] = W[tid, :].clone()
            W[tid, :] = vec.to(W.dtype, non_blocking=True)

    # ---------------- 이미지 생성: log_validation ----------------
    # log_validation은 Accelerator를 요구 → 그대로 전달
    run_step = int(500)  # 파일 구분용 고유 step
    comment  = f"snr{cur_snr:g}"

    _run_log_validation(
        logger=logger,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        args=args,
        accelerator=accelerator,
        weight_dtype=weight_dtype,
        global_step=run_step,
        prompt=val_prompt,
        comment=comment,   # "clean" 아님 → 노이즈 경로 사용
        subpath="generated_images",
    )

    # ---------------- 방금 생성한 파일만 수집 ----------------
    # 파일명: step-{run_step}_{comment}_{i}.jpeg
    step_pat = re.compile(rf"^step-{run_step}_[^_]+_\d+\.(?:png|jpe?g|tiff)$", re.IGNORECASE)
    if not val_dir.exists():
        logger.warning(f"[metrics] validation_images not found: {val_dir}")
        # 원복 & args 복원
        with torch.no_grad():
            for tid, vec in backup.items(): W[tid, :] = vec
        if orig_num_validation_images: setattr(args, "num_validation_images", orig_num_validation_images)
        setattr(args, "text_snr_db", orig_text_snr_db)
        if orig_text_channel_type is not None: setattr(args, "text_channel_type", orig_text_channel_type)
        setattr(args, "placeholder_token", orig_placeholder_token)
        return {}

    gen_paths = [str(p) for p in val_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".tiff"} and step_pat.match(p.name)]
    gen_paths = sorted(gen_paths)
    if not gen_paths:
        logger.warning(f"[metrics] No generated images found for step={run_step}.")
        with torch.no_grad():
            for tid, vec in backup.items(): W[tid, :] = vec
        if orig_num_validation_images: setattr(args, "num_validation_images", orig_num_validation_images)
        setattr(args, "text_snr_db", orig_text_snr_db)
        if orig_text_channel_type is not None: setattr(args, "text_channel_type", orig_text_channel_type)
        setattr(args, "placeholder_token", orig_placeholder_token)
        return {}

    # ---------------- 레퍼런스 이미지 ----------------
    td = Path(args.train_data_dir)
    ref_name = getattr(args, "ref_filename", "00.jpg")
    ref_file = td / ref_name
    if not ref_file.exists():
        logger.warning(f"[metrics] Reference image '{ref_name}' not found in {td}.")
        with torch.no_grad():
            for tid, vec in backup.items(): W[tid, :] = vec
        if orig_num_validation_images: setattr(args, "num_validation_images", orig_num_validation_images)
        setattr(args, "text_snr_db", orig_text_snr_db)
        if orig_text_channel_type is not None: setattr(args, "text_channel_type", orig_text_channel_type)
        setattr(args, "placeholder_token", orig_placeholder_token)
        return {}
    ref_paths = [str(ref_file)]

    # ---------------- 메트릭 계산 ----------------
    # (모델들은 Accelerator로 wrap하지 않고 device만 맞춰 평가)
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False); clip_model.eval()
    dino_model = ViTModel.from_pretrained("facebook/dino-vits16").to(device).eval()
    dream_model, dream_preproc = dreamsim(pretrained=True, device=device)
    lpips_model = LPIPS(net="vgg").to(device).eval()

    # CLIP
    clip_gen = extract_all_clip_images(gen_paths, clip_model, device, batch_size=min(64, len(gen_paths)), num_workers=2)
    clip_ref = extract_all_clip_images(ref_paths, clip_model, device, batch_size=len(ref_paths), num_workers=2)
    clip_gen = clip_gen / np.clip(np.linalg.norm(clip_gen, axis=1, keepdims=True), 1e-8, None)
    clip_ref = clip_ref / np.clip(np.linalg.norm(clip_ref, axis=1, keepdims=True), 1e-8, None)
    sims = clip_gen @ clip_ref.T
    clip_best_per_gen = sims.max(axis=1)
    clip_best_idx = int(np.argmax(clip_best_per_gen))
    clip_best_score = float(clip_best_per_gen[clip_best_idx])

    # DINO
    dino_gen = extract_all_dino_images(gen_paths, dino_model, device, batch_size=min(64, len(gen_paths)), num_workers=2)
    dino_ref = extract_all_dino_images(ref_paths, dino_model, device, batch_size=len(ref_paths), num_workers=2)
    dino_gen = dino_gen / np.clip(np.linalg.norm(dino_gen, axis=1, keepdims=True), 1e-8, None)
    dino_ref = dino_ref / np.clip(np.linalg.norm(dino_ref, axis=1, keepdims=True), 1e-8, None)
    dino_sims = dino_gen @ dino_ref.T
    dino_best_per_gen = dino_sims.max(axis=1)
    dino_best_idx = int(np.argmax(dino_best_per_gen))
    dino_best_score = float(dino_best_per_gen[dino_best_idx])

    # DreamSim (similarity = 1 - distance)
    dream_best_per_gen = dreamsim_best_per_gen(gen_paths, ref_paths, dream_model, dream_preproc,
                                               device=device, batch_size=16)
    dream_best_idx = int(np.argmax(dream_best_per_gen))
    dream_best_score = float(dream_best_per_gen[dream_best_idx])

    # LPIPS (distance ↓)
    lpips_best_dists = lpips_best_per_gen(gen_paths, ref_paths, lpips_model, device=device, batch_size=16)
    lpips_best_idx = int(np.argmin(lpips_best_dists))
    lpips_best_dist = float(lpips_best_dists[lpips_best_idx])
    lpips_best_sim  = float(1.0 / (1.0 + lpips_best_dist))

    # ---------------- 베스트 이미지 저장 ----------------
    clip_best_path  = str(best_dir / "best_clip.png")
    dino_best_path  = str(best_dir / "best_dino.png")
    dream_best_path = str(best_dir / "best_dreamsim.png")
    lpips_best_path = str(best_dir / "best_lpips.png")
    shutil.copyfile(gen_paths[clip_best_idx],  clip_best_path)
    shutil.copyfile(gen_paths[dino_best_idx],  dino_best_path)
    shutil.copyfile(gen_paths[dream_best_idx], dream_best_path)
    shutil.copyfile(gen_paths[lpips_best_idx], lpips_best_path)

    results = {
        "clip":      {"best_score": clip_best_score,   "best_image": clip_best_path},
        "dino":      {"best_score": dino_best_score,   "best_image": dino_best_path},
        "dreamsim":  {"best_score": dream_best_score,  "best_image": dream_best_path},
        "lpips":     {"best_distance": lpips_best_dist,
                      "best_similarity": lpips_best_sim,
                      "best_image": lpips_best_path},
        "mean": {
            "clip": float(sims.mean()),
            "dino": float(dino_sims.mean()),
            "dreamsim": float(dream_best_per_gen.mean()),
            "lpips": float(lpips_best_dists.mean()),
        },
        "meta": {
            "snr_db": cur_snr,
            "channel_type": getattr(args, "channel_type", "none"),
            "used_step": run_step,
            "generated_count": len(gen_paths),
            "output_dir": str(args.output_dir),
            "ref_image": ref_name,
        }
    }

    logger.info(
        f"[metrics] SNR={cur_snr}dB step={run_step}  "
        f"BEST→  CLIP:{clip_best_score:.4f}  DINO:{dino_best_score:.4f}  "
        f"DreamSim:{dream_best_score:.4f}  LPIPS↓:{lpips_best_dist:.4f}"
    )

    # ---------------- 원복 ----------------
    with torch.no_grad():
        for tid, vec in backup.items():
            W[tid, :] = vec

    if orig_num_validation_images:
        setattr(args, "num_validation_images", orig_num_validation_images)
    setattr(args, "text_snr_db", orig_text_snr_db)
    if orig_text_channel_type is not None:
        setattr(args, "text_channel_type", orig_text_channel_type)
    setattr(args, "placeholder_token", orig_placeholder_token)

    return results
