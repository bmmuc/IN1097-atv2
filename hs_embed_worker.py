import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

MODEL_IDS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "hatebert": "GroNLP/hateBERT",
    "mbert": "google-bert/bert-base-multilingual-cased",
}

BATCH_SIZE = 64
MAX_LENGTH = 128


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_minilm(texts, device, fp16):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_IDS["minilm"], device=device)
    if fp16 and device == "cuda":
        model = model.half()
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embs, dtype=np.float32)


def encode_transformer(texts, model_tag, device, fp16):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = MODEL_IDS[model_tag]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    if fp16 and device == "cuda":
        model = model.half()
    model.eval()

    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            out_chunks.append(pooled.float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0).astype(np.float32)


def encode(texts, model_tag, device, fp16):
    if model_tag == "minilm":
        return encode_minilm(texts, device, fp16)
    return encode_transformer(texts, model_tag, device, fp16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS.keys()))
    ap.add_argument("--datasets-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-ids", default="", help="comma-separated subset; empty = all parquet files found")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and device == "cuda"

    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.datasets_dir, "*.parquet")))
    if args.dataset_ids:
        wanted = set(args.dataset_ids.split(","))
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in wanted]

    print(f"[hs_embed_worker] model={args.model} device={device} fp16={fp16} n_datasets={len(paths)}", flush=True)

    t0 = time.time()
    n_done = 0
    n_skipped = 0
    for path in paths:
        dataset_id = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out_dir, f"{dataset_id}__{args.model}.npy")
        if os.path.exists(out_path):
            n_skipped += 1
            continue

        df = pd.read_parquet(path, columns=["text", "label"])
        texts = df["text"].astype(str).tolist()

        t1 = time.time()
        embs = encode(texts, args.model, device, fp16)
        dt = time.time() - t1

        tmp_path = out_path + ".tmp.npy"
        np.save(tmp_path, embs)
        os.replace(tmp_path, out_path)
        n_done += 1
        print(f"[hs_embed_worker] {dataset_id} n={len(texts)} dim={embs.shape[1]} {dt:.1f}s", flush=True)

    print(
        f"[hs_embed_worker] model={args.model} done: encoded={n_done} skipped={n_skipped} "
        f"total_time={time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
