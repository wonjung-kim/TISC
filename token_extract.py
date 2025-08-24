from dataclasses import dataclass
from typing import List, Tuple, Dict, Literal, Optional
import os, json, math, numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoProcessor, BlipForConditionalGeneration, BlipProcessor

# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    model_name: str = "Salesforce/blip-image-captioning-large"
    # generation
    max_new_tokens: int = 32
    num_beams: int = 1
    top_p: float = 0.9
    seed: int = 42
    do_sample: bool = True  # Enable sampling for more diverse captions
    # dataset
    dataset_id: str = "google/dreambooth"
    TARGET_CONFIGS: List[str] | None = None  # e.g., ["wolf_plushie", "dog6"]
    max_images_per_config: Optional[int] = None # None = use all
    # selection
    EXCLUDE_PROMPT_FROM_TOPK: bool = True
    PROMPT: str | None = "a photo of" # e.g., "a photography of"
    MODE: Literal["attn_scr","cross_scr"] = "attn_scr"  # self-attn vs cross-attn scoring
    TOP_Tok: int = 3
    use_last_k_layers: int = 7 # use last k layers for scoring
    pair_pool: Literal["mean","max"] = "max"  # pooling over image tokens for cross-attn
    exclude_punct: bool = True

CFG = CFG()
torch.manual_seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)

# =========================
# SETUP: Load model and processor
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(CFG.model_name)
model = BlipForConditionalGeneration.from_pretrained(CFG.model_name).to(device)
model.eval()

# make sure model can output attentions
model.config.output_attentions = True
model.config.output_hidden_states = True
if hasattr(model, "text_decoder") and hasattr(model.text_decoder, "config"):
    model.text_decoder.config.output_attentions = True
    model.text_decoder.config.output_hidden_states = True
if hasattr(model, "vision_model") and hasattr(model.vision_model, "config"):
    model.vision_model.config.output_attentions = False
print(f"Using model: {CFG.model_name} on device: {device}")

# Special token ids for filtering
SPECIAL_IDS = set([processor.tokenizer.cls_token_id, processor.tokenizer.sep_token_id, processor.tokenizer.pad_token_id,
    getattr(processor.tokenizer, "bos_token_id", None), getattr(processor.tokenizer, "eos_token_id", None),])
SPECIAL_IDS.discard(None)
PUNCT = set(list(",.;:!?"))

# =========================
# DATA: iterate DreamBooth images
# =========================
all_configs = CFG.TARGET_CONFIGS or get_dataset_config_names(CFG.dataset_id)
print("Dreambooth Configs:", all_configs)

def iter_images():
    for name in all_configs:
        ds = load_dataset(CFG.dataset_id, name, split="train")
        limit = CFG.max_images_per_config
        n = len(ds) if limit is None else min(len(ds), int(limit))
        subset = ds if limit is None else ds.select(range(n))
        for i, ex in enumerate(subset):
            yield name, i, ex["image"]

# =========================
# TOKEN/WORD MAPPING UTILS
# =========================
def decode(ids: torch.Tensor) -> str:
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

# These functions extract spans of word tokens from the tokenized input. (Token -> Word Merging)
def _spans_from_word_ids(enc):
    offsets = enc["offset_mapping"]
    wids = enc.word_ids()
    spans = []
    curr = None
    for t_idx, w in enumerate(wids):
        if w is None: continue
        s, e = offsets[t_idx]
        if curr is None:
            curr = [w, s, e]
        elif w == curr[0]:
            curr[2] = e
        else:
            spans.append((curr[1], curr[2]))
            curr = [w, s, e]
    if curr is not None:
        spans.append((curr[1], curr[2]))
    return spans

def word_level_groups(text: str) -> Tuple[List[str], List[List[int]], Dict[int, Tuple[int,int]]]:
    enc = processor.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    toks = processor.tokenizer.convert_ids_to_tokens(enc["input_ids"])  # wordpieces (unused)
    wids = enc.word_ids()
    groups: Dict[int, List[int]] = {}
    for t_idx, w in enumerate(wids):
        if w is None:  # spaces
            continue
        groups.setdefault(w, []).append(t_idx)
    words = [text[s:e] for (s, e) in _spans_from_word_ids(enc)]
    token_groups = [groups.get(i, []) for i in range(len(words))]
    spans = {i: (s, e) for i, (s, e) in enumerate(_spans_from_word_ids(enc))}
    return words, token_groups, spans

# calculate prompt lenth to exclude it from attention scoring
def count_prompt_words_in_caption(prompt: str | None, caption: str) -> int:
    if not prompt:
        return 0
    p = prompt.strip()
    c = caption.lstrip()
    if not c.lower().startswith(p.lower()):
        return 0
    enc_p = processor.tokenizer(p, return_offsets_mapping=True, add_special_tokens=False)
    wids = enc_p.word_ids()
    n_words = len({w for w in wids if w is not None})
    return n_words

def replace_spans(text: str, spans: List[Tuple[int,int]], repl: str = "*") -> str:
    # Replace spans without shifting indices: go from right to left
    s = text
    for st, en in sorted(spans, key=lambda x: x[0], reverse=True):
        s = s[:st] + repl + s[en:]
    return s

# =========================
# FORWARD PASSES (Generate caption, Teacher-forced pass for attention extraction)
# =========================
@torch.no_grad()
def generate_caption(pil_img: Image.Image) -> Tuple[str, torch.Tensor]:
    if CFG.PROMPT:  # conditional captioning
        inputs = processor(images=pil_img, text=CFG.PROMPT, return_tensors="pt").to(device)
        gen = model.generate(
            pixel_values=inputs.pixel_values,
            input_ids=inputs.input_ids, 
            max_new_tokens=CFG.max_new_tokens,
            num_beams=CFG.num_beams,
            do_sample=CFG.do_sample,
            top_p=CFG.top_p,
            return_dict_in_generate=True,
        )
    else:
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        gen = model.generate(
            pixel_values=inputs.pixel_values,
            max_new_tokens=CFG.max_new_tokens,
            num_beams=CFG.num_beams,
            do_sample=CFG.do_sample,
            top_p=CFG.top_p,
            return_dict_in_generate=True,
        )
    caption = decode(gen.sequences)
    return caption, inputs.pixel_values

@torch.no_grad()
def teacher_forced_pass(pixel_values: torch.Tensor, caption: str):
    tok = processor.tokenizer(caption, return_tensors="pt").to(device)
    input_ids = tok.input_ids
    # -------- 1) top-level: get decoder's attention directly --------
    out = model(
        pixel_values=pixel_values.to(device),
        input_ids=input_ids,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )
    dec_self  = getattr(out, "decoder_attentions", None)
    dec_cross = getattr(out, "cross_attentions", None)
    if dec_self is not None and len(dec_self) > 0:
        self_atts  = [a[0].detach().cpu() for a in dec_self]             # [L][H,T,T]
        cross_atts = [a[0].detach().cpu() for a in (dec_cross or [])]    # [L][H,T,S]
        return input_ids[0].detach().cpu(), self_atts, cross_atts
    # -------- 2) fallback: vision encoder → text decoder --------
    vout = model.vision_model(
        pixel_values=pixel_values.to(device),
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    enc = vout.last_hidden_state                     # [B,S,H], S=이미지 패치 토큰 수
    enc_mask = torch.ones(enc.shape[:2], dtype=torch.long, device=device)  # ViT는 패딩 없음
    txt_mask = torch.ones_like(input_ids, device=device)
    dec_out = model.text_decoder(
        input_ids=input_ids,
        attention_mask=txt_mask,
        encoder_hidden_states=enc,
        encoder_attention_mask=enc_mask,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    self_atts  = [a[0].detach().cpu() for a in (dec_out.attentions or [])]          # [L][H,T,T]
    cross_atts = [a[0].detach().cpu() for a in (dec_out.cross_attentions or [])]    # [L][H,T,S]
    return input_ids[0].detach().cpu(), self_atts, cross_atts

# =========================
# SCORING FUNCTIONS
# =========================
def strip_special_by_ids(token_scores: np.ndarray, input_ids: torch.Tensor) -> np.ndarray:
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 2:
            ids = input_ids[0].tolist()
        elif input_ids.ndim == 1:
            ids = input_ids.tolist()
        else:
            raise ValueError(f"Unexpected input_ids shape: {tuple(input_ids.shape)}")
    else:
        # list-like fallback
        ids = list(input_ids)
    keep = [i for i, tid in enumerate(ids) if tid not in SPECIAL_IDS]
    token_scores = np.asarray(token_scores)
    return token_scores[keep]

def _trim_to_no_special(scores: np.ndarray, caption: str) -> np.ndarray:
    # Retokenize without specials to align with wordpiece grouping
    no_spec = processor.tokenizer(caption, return_tensors="pt", add_special_tokens=False)
    T_no = no_spec.input_ids.shape[1]
    if len(scores) == T_no:
        return scores
    # If scores contain specials (likely CLS + SEP), drop both ends to match
    if len(scores) >= T_no + 2:
        return scores[1:1+T_no]
    # Otherwise, fallback: pad/truncate to T_no
    out = np.zeros(T_no, dtype=float)
    m = min(T_no, len(scores))
    out[:m] = scores[:m]
    return out

def score_self_attention(self_atts: List[torch.Tensor], last_k: int, q_exclude_first: int = 0) -> np.ndarray:
    # Average attention INTO each token across last-k layers, heads, and all query positions
    chosen = self_atts[-last_k:] if last_k > 0 else self_atts
    if not chosen:
        return np.zeros(0, dtype=float)
    
    A = torch.stack(chosen)  # [K,H,T_q,T_k]
    # Exclude prompt tokens from query attention
    if q_exclude_first > 0:
        A = A[:, :, q_exclude_first:, :]  # [K,H,T_q',T_k]
    # Restrain sink attentions
    A = A - A.mean(dim=-1, keepdim=True)  # center over keys
    A = torch.clamp(A, min=0)
    scores = A.mean(dim=(0, 1, 2))  # [T_k]
    return scores.numpy()

def score_cross_attention(cross_atts: List[torch.Tensor], last_k: int, pool: str = "mean") -> np.ndarray:
    # For each text token, aggregate its cross-attn over image tokens
    chosen = cross_atts[-last_k:] if last_k > 0 else cross_atts
    if not chosen:
        return np.zeros(0, dtype=float)
    stacked = torch.stack(chosen)  # [K,H,T,S]
    # avg over layers, heads first => [T,S]
    avg = stacked.mean(dim=(0,1))
    if pool == "max":
        per_tok = avg.max(dim=-1).values  # [T]
    else:
        per_tok = avg.mean(dim=-1)        # [T]
    return per_tok.numpy()

def topk_words_by_scores(caption: str, token_scores: np.ndarray, k: int, input_ids_with_specials: torch.Tensor, exclude_punct: bool = True, exclude_first_n_words: int = 0,):
    # Map token scores -> word scores (sum over wordpieces)
    token_scores = strip_special_by_ids(token_scores, input_ids_with_specials)
    words, token_groups, spans = word_level_groups(caption)
    
    word_scores = []
    for w_idx, grp in enumerate(token_groups):
        if w_idx < exclude_first_n_words or not grp:
            word_scores.append(float("-inf")); continue
        w_text = words[w_idx].strip()
        if exclude_punct and (not w_text or set(w_text) <= PUNCT):
            word_scores.append(float("-inf")); continue
        word_scores.append(float(np.sum(token_scores[grp])))
    
    order = np.argsort(word_scores)[::-1]
    topk, seen = [], 0
    for w in order:
        if word_scores[w] == float("-inf"):
            continue
        topk.append((int(w), spans[w], float(word_scores[w]), words[w]))  # ← 텍스트 포함
        seen += 1
        if seen >= k:
            break
    return topk  # (w_idx, (start,end), score, word_text)

# =========================
# MAIN LOOP
# =========================
results = []
for cfg_name, idx, img in iter_images():
    # Generate caption and extract attention scores
    caption, px = generate_caption(img)
    input_ids, self_atts, cross_atts = teacher_forced_pass(px, caption)

    # Calculate token scores based on the selected mode
    if CFG.MODE == "attn_scr":
        tok_scores = score_self_attention(self_atts, CFG.use_last_k_layers,
                                      q_exclude_first=count_prompt_words_in_caption(CFG.PROMPT, caption) if CFG.EXCLUDE_PROMPT_FROM_TOPK else 0)
    elif CFG.MODE == "cross_scr":
        tok_scores = score_cross_attention(cross_atts, CFG.use_last_k_layers, CFG.pair_pool)
    else:
        raise ValueError("Unknown MODE: use 'attn_scr' or 'cross_scr'")

    # Exclude prompt words from top-k selection if configured
    exclude_n = count_prompt_words_in_caption(CFG.PROMPT, caption) if CFG.EXCLUDE_PROMPT_FROM_TOPK else 0
    topk = topk_words_by_scores(caption, tok_scores, CFG.TOP_Tok,input_ids_with_specials=input_ids,
        exclude_punct=CFG.exclude_punct, exclude_first_n_words=exclude_n)
    spans = [sp for _, sp, _, _ in topk] 
    
    # Change caption to a starred version
    starred = replace_spans(caption, spans, "*")
    rec = {
        "config": cfg_name,
        "index": idx,
        "caption": caption,
        "starred": starred,
        "mode": CFG.MODE,
        "pair_pool": CFG.pair_pool,
        "selected": [
            {"word_idx": w, "span": s, "score": round(sc, 6), "word": txt}
            for (w, s, sc, txt) in topk
        ],
    }

    # Print log
    sel_str = ", ".join(
        f"(i={w}, word={json.dumps(txt, ensure_ascii=False)}, span={s}, score={sc:.4f})"
        for (w, s, sc, txt) in topk
    ) if topk else ""
    log = (
        f"[{cfg_name}/{idx}] mode={CFG.MODE}\n"
        f"  caption: {caption}\n"
        f"  starred: {starred}\n"
        f"  selected: [{sel_str}]"
    )
    print(log + "\n")

    # Deubugging: print token scores
    # if self_atts:
    #     H, T, _ = self_atts[-1].shape  # [H,T,T]
    #     log += f"\n  self_attn_last: H={H}, T={T}"
    # if cross_atts:
    #     H2, T2, S = cross_atts[-1].shape  # [H,T,S]
    #     log += f"\n  cross_attn_last: H={H2}, T={T2}, S={S}"
    # print(log + "\n")
    # Deubugging Finished
    
    # Save results
    results.append(rec)

output = f"/root/cha/wjkim/diffusers/examples/neurips/blip_generated_prompt/dream_blip_{CFG.MODE}_{CFG.TOP_Tok}.jsonl"
with open(output, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Saved files (", len(results), ")")
