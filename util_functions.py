import os
import torch
from diffusers import DiffusionPipeline
from pathlib import Path
import json

# def log_validation(logger, text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, prompt, comment=None):
#     logger.info(
#         f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
#         f" {prompt}."
#     )
#     # create pipeline (note: unet and vae are loaded again in float32)
#     pipeline = DiffusionPipeline.from_pretrained(
#         args.pretrained_model_name_or_path,
#         text_encoder=accelerator.unwrap_model(text_encoder),
#         tokenizer=tokenizer,
#         unet=unet,
#         vae=vae,
#         revision=args.revision,
#         torch_dtype=weight_dtype,
#     )
#     # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
#     pipeline = pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=True)

#     # run inference
#     generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)



#     validation_path = os.path.join(args.output_dir, "validation_images")
#     os.makedirs(validation_path, exist_ok=True)
#     for i in range(args.num_validation_images):
#         with torch.autocast("cuda"):
#             image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5,
#                              generator=generator).images[0]
#             if comment is not None:
#                 validation_image_path = f"{validation_path}/step-{global_step}_{comment}_{i + 1}.jpeg"
#             else:
#                 validation_image_path = f"{validation_path}/step-{global_step}_{i + 1}.jpeg"
#             image.save(validation_image_path)

#     del pipeline
#     torch.cuda.empty_cache()

def log_validation(logger, text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, prompt, comment=None, subpath="validation_images"):
    import math
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt: {prompt}."
    )

    # ---------- helper: add wireless noise to selected embedding vectors ----------
    def _add_wireless_noise_to_vectors(W: torch.Tensor, channel_type: str, snr_db: float, rng: torch.Generator):
        if channel_type == "none":
            return W
        single = (W.dim() == 1)
        if single:
            W = W.unsqueeze(0)  # [1,H]

        eps = torch.finfo(W.dtype).eps
        P = torch.clamp(W.pow(2).mean().detach(), min=eps)
        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        sigma = torch.sqrt(torch.clamp(P / (snr_lin + eps), min=eps)).to(W.dtype)

        if channel_type.lower() == "awgn":
            noise = torch.randn(W.shape, device=W.device, dtype=W.dtype, generator=rng) * sigma
            Y = W + noise
        elif channel_type.lower() == "rayleigh":
            K = W.size(0)
            z1 = torch.randn((K, 1), device=W.device, dtype=W.dtype, generator=rng)
            z2 = torch.randn((K, 1), device=W.device, dtype=W.dtype, generator=rng)
            h = torch.sqrt(z1**2 + z2**2) / math.sqrt(2.0)
            y_sig = W * h
            noise = torch.randn(W.shape, device=W.device, dtype=W.dtype, generator=rng) * sigma
            Y = y_sig + noise
        else:
            Y = W
        return Y.squeeze(0) if single else Y

    # ---------- build pipeline ----------
    te_raw = accelerator.unwrap_model(text_encoder)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=te_raw,
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # ---------- (optional) temporarily noise-inject the placeholder embedding(s) ----------
    placeholder_token = getattr(args, "placeholder_token", None)
    channel_type = getattr(args, "text_channel_type", getattr(args, "channel_type", "none"))
    snr_db = float(getattr(args, "text_snr_db", getattr(args, "snr_db", 20.0)))
    rng = torch.Generator(device=accelerator.device).manual_seed(int(getattr(args, "text_embed_noise_seed", 0)))

    # NEW: clean 모드면 주입 비활성화
    inject_noise = (comment != "clean") and (placeholder_token is not None) and (channel_type != "none")
    logger.info(f"Token: {placeholder_token}, Channel: {channel_type}, SNR: {snr_db} dB, inject_noise={inject_noise}")

    orig_vectors = None
    placeholder_ids = None

    if inject_noise:
        ids = tokenizer.convert_tokens_to_ids(placeholder_token)
        if ids is None:
            logger.warning(f"[log_validation] placeholder_token '{placeholder_token}' not found in tokenizer. Skipping embed noise.")
        else:
            if isinstance(ids, int):
                placeholder_ids = [ids]
            elif isinstance(ids, (list, tuple)):
                placeholder_ids = [i for i in ids if isinstance(i, int) and i >= 0]
            else:
                placeholder_ids = []

    try:
        if inject_noise and placeholder_ids:
            emb = te_raw.get_input_embeddings()
            W = emb.weight.data
            valid_ids = [i for i in placeholder_ids if 0 <= i < W.size(0)]
            if len(valid_ids) == 0:
                logger.warning("[log_validation] No valid placeholder ids to noise.")
            else:
                with torch.no_grad():
                    orig_vectors = W[valid_ids, :].clone()
                    noisy = _add_wireless_noise_to_vectors(W[valid_ids, :], channel_type, snr_db, rng)
                    W[valid_ids, :] = noisy

        # ---------- run inference ----------
        generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
        validation_path = os.path.join(args.output_dir, subpath)
        os.makedirs(validation_path, exist_ok=True)

        for i in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
            if comment is not None:
                validation_image_path = f"{validation_path}/step-{global_step}_{comment}_{i + 1}.jpeg"
            else:
                validation_image_path = f"{validation_path}/step-{global_step}_{i + 1}.jpeg"
            image.save(validation_image_path)

    finally:
        # ---------- restore original embeddings ----------
        if inject_noise and (placeholder_ids and orig_vectors is not None):
            emb = te_raw.get_input_embeddings()
            with torch.no_grad():
                emb.weight.data[placeholder_ids, :] = orig_vectors
        del pipeline
        torch.cuda.empty_cache()


def _parse_placeholders(s):
    if s is None:
        return []
    return [t.strip() for t in str(s).split(",") if t.strip()]

def save_progress(weight_vector, accelerator, args, save_path):
    """
    멀티 토큰 호환 save:
    - .bin : torch.save(payload)  → 로드/재학습·디버깅에 유용
    - .json: eval.py가 바로 읽을 수 있게 'weight'/'weights' 키로 [M,768] 저장
    포함 메타:
      * placeholders (리스트)
      * num_vectors_per_token (V), num_placeholders (P), embedding_dim (D)
      * group_slices: [(s,e), ...]  (각 placeholder가 차지하는 W 구간)
      * composed_embeddings: [P,D] (그룹 평균; 검증에서 베이스 토큰에 주입할 때 편함)
      * (가능하면) placeholder_id_groups, placeholder_base_ids
    """
    model = accelerator.unwrap_model(weight_vector)
    W = model.weight.detach().cpu()                  # [P*V, D]
    D = int(W.shape[-1])

    # V(토큰당 벡터수) / P(플레이스홀더 개수)
    V = int(getattr(model, "num_vectors_per_token",
                    getattr(args, "num_vectors_per_token", 1)))
    G = int(W.shape[0])                              # 총 벡터 개수 = P*V
    if V <= 0:
        V = 1
    P = max(G // V, 1)

    placeholders = _parse_placeholders(getattr(args, "placeholder_token", None))
    if not placeholders:
        # 안전장치(메타 용도)
        placeholders = [str(getattr(args, "placeholder_token", "<v*>"))]
    # 길이 불일치는 허용(메타); group_slices로 복원 가능

    # 그룹 평균(간단 조합). 학습 모듈 조합 방식과 다르면 여기 로직을 맞추면 됨.
    composed_list = []
    group_slices = []
    for i in range(P):
        s = i * V
        e = min(s + V, G)
        group_slices.append((int(s), int(e)))
        composed_list.append(W[s:e].mean(dim=0))     # [D]
    composed = torch.stack(composed_list, dim=0)     # [P, D]

    # torch 저장 payload
    payload = {
        "version": 2,
        "placeholder_token": getattr(args, "placeholder_token", None),  # 하위호환 키
        "learned_embedding": W,                  # [P*V, D] (텐서)
        "placeholders": placeholders,            # 리스트
        "num_placeholders": P,
        "num_vectors_per_token": V,
        "embedding_dim": D,
        "group_slices": group_slices,            # 리스트[(s,e)]
        "composed_embeddings": composed,         # [P, D] (텐서)
    }
    if hasattr(args, "initializer_token"):
        payload["initializer_tokens"] = _parse_placeholders(args.initializer_token)
    if hasattr(model, "placeholder_id_groups"):
        payload["placeholder_id_groups"] = [
            [int(x) for x in group] for group in model.placeholder_id_groups
        ]
    if hasattr(model, "placeholder_base_ids"):
        payload["placeholder_base_ids"] = [int(x) for x in model.placeholder_base_ids]

    # 1) .bin 저장 (torch.save)
    save_path = Path(f'{save_path}.bin')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(save_path))

    # 2) .json 저장 (eval.py 호환: 'weight'/'weights' 둘 다 제공)
    json_path = save_path.with_suffix(".json") if save_path.suffix else Path(str(save_path) + ".json")
    json_payload = {
        "version": 2,
        # eval.py에서 우선적으로 찾는 키 이름들
        "weights": W.tolist(),
        # 메타(참고용)
        "placeholders": placeholders,
        "num_placeholders": P,
        "num_vectors_per_token": V,
        "embedding_dim": D,
        "group_slices": group_slices,
        "composed_embeddings": composed.tolist(),
        "placeholder_token": getattr(args, "placeholder_token", None),
        "initializer_tokens": _parse_placeholders(getattr(args, "initializer_token", None)),
    }
    # 선택: id 메타(JSON 직렬화 가능 형태로만)
    if "placeholder_id_groups" in payload:
        json_payload["placeholder_id_groups"] = payload["placeholder_id_groups"]
    if "placeholder_base_ids" in payload:
        json_payload["placeholder_base_ids"] = payload["placeholder_base_ids"]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    # (선택) 경로 반환하면 호출부에서 로그 찍기 쉬움
    return {"bin_path": str(save_path), "json_path": str(json_path)}

# def save_progress(weight_vector, accelerator, args, save_path):
#     """
#     멀티 토큰 호환 save:
#     - learned_embedding: raw 스택 형태 [P*V, D] (P=플레이스홀더 개수, V=토큰당 벡터수)
#     - composed_embeddings: 그룹 평균 [P, D] (로드 시 베이스 토큰에 바로 주입할 때 유용)
#     - placeholders, num_vectors_per_token 등 메타 저장
#     - (있으면) placeholder_id_groups / placeholder_base_ids도 함께 저장
#     """
#     model = accelerator.unwrap_model(weight_vector)
#     W = model.weight.detach().cpu()              # [P*V, D]
#     D = W.shape[-1]

#     # 벡터 수/플레이스홀더 수 추정
#     V = int(getattr(model, "num_vectors_per_token",
#                     getattr(args, "num_vectors_per_token", 1)))
#     G = W.shape[0]                               # 총 벡터 개수 = P*V
#     P = G // V if V > 0 else 1                   # 플레이스홀더 개수

#     placeholders = _parse_placeholders(getattr(args, "placeholder_token", None))
#     if not placeholders:
#         # 안전장치: args가 리스트 형식이 아니거나 비어있을 때
#         placeholders = [str(getattr(args, "placeholder_token", "<v*>"))]
#     # 길이 불일치 시 최대한 맞춰 잘라내거나 패딩하지 않고 남긴다(메타로만 사용)
#     if len(placeholders) != P:
#         # 길이가 다르더라도 저장은 진행. 로더가 group_slices 기준으로 사용 가능.
#         pass

#     # 그룹 평균(조합) 임베딩 산출: 여기서는 간단히 평균
#     composed_list = []
#     group_slices = []  # [(start, end), ...] for each placeholder
#     for i in range(P):
#         s = i * V
#         e = min(s + V, G)
#         group_slices.append((s, e))
#         composed_list.append(W[s:e].mean(dim=0))     # [D]
#     composed = torch.stack(composed_list, dim=0)     # [P, D]

#     payload = {
#         "version": 2,
#         # 기존 키(하위호환)
#         "placeholder_token": getattr(args, "placeholder_token", None),
#         "learned_embedding": W,                 # [P*V, D]
#         # 추가 메타
#         "placeholders": placeholders,           # 리스트화된 플레이스홀더
#         "num_placeholders": P,
#         "num_vectors_per_token": V,
#         "embedding_dim": D,
#         "group_slices": group_slices,          # 각 플레이스홀더가 차지하는 구간
#         "composed_embeddings": composed,       # [P, D] (베이스 토큰에 주입용)
#     }

#     # 선택적: 초기화 토큰 정보도 함께 저장(있을 때)
#     if hasattr(args, "initializer_token"):
#         payload["initializer_tokens"] = _parse_placeholders(args.initializer_token)

#     # 선택적: 임베딩-토큰ID 매핑(모델이 보유하고 있다면)
#     if hasattr(model, "placeholder_id_groups"):
#         payload["placeholder_id_groups"] = [
#             [int(x) for x in group] for group in model.placeholder_id_groups
#         ]
#     if hasattr(model, "placeholder_base_ids"):
#         payload["placeholder_base_ids"] = [int(x) for x in model.placeholder_base_ids]

#     torch.save(payload, save_path)