#!/usr/bin/env python3
import os
import re
import time
import pickle
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis


def slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:120] or "actor"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n


def pick_largest_face(faces) -> Optional[object]:
    if not faces:
        return None
    def area(f):
        b = f.bbox  # [x1,y1,x2,y2]
        return float((b[2] - b[0]) * (b[3] - b[1]))
    return max(faces, key=area)


def iter_images(folder: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def create_actor_pkls(
    dataset_folder: str,
    output_dir: str,
    model_pack: str = "buffalo_l",
    det_size: Tuple[int, int] = (640, 640),
    gpu_id: int = 0,
    shard_id: int = 0,
    num_shards: int = 1,
) -> None:
    ds = Path(dataset_folder)
    outd = Path(output_dir)
    outd.mkdir(parents=True, exist_ok=True)

    if not ds.exists():
        raise FileNotFoundError(f"Dataset folder not found: {ds}")

    # IMPORTANT: pin this process to one GPU (optional but recommended in multi-GPU servers)
    # If you want to run CPU: set gpu_id=-1
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        ctx_id = 0  # inside this process, visible GPU becomes ctx_id=0
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        ctx_id = -1
        providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(name=model_pack, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)

    people_folders = [p for p in ds.iterdir() if p.is_dir()]
    people_folders = sorted(people_folders, key=lambda p: p.name.lower())

    # Optional sharding across multiple processes
    # Example: run 2 processes with shard_id 0/1, num_shards 2
    people_folders = [p for i, p in enumerate(people_folders) if (i % num_shards) == shard_id]

    print(f"Dataset:   {ds}")
    print(f"Output:    {outd}")
    print(f"Model:     {model_pack}")
    print(f"Det size:  {det_size}")
    print(f"GPU:       {gpu_id}  (ctx_id={ctx_id})")
    print(f"Sharding:  shard_id={shard_id}, num_shards={num_shards}")
    print(f"People:    {len(people_folders)} folders")

    total_written = 0
    total_skipped = 0

    for person_dir in tqdm(people_folders, desc="Actors"):
        person_name = person_dir.name
        images = iter_images(person_dir)

        if not images:
            total_skipped += 1
            continue

        embs = []
        used = 0
        for img_path in images:
            try:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue

                faces = app.get(bgr)
                best = pick_largest_face(faces)
                if best is None:
                    continue

                # Use normalized embedding for cosine similarity / inner-product FAISS
                emb = best.normed_embedding.astype(np.float32)
                embs.append(emb)
                used += 1
            except Exception:
                continue

        if not embs:
            total_skipped += 1
            continue

        # Aggregate multiple images -> one robust actor embedding
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        mean_emb = l2_normalize(mean_emb).astype(np.float32)

        payload = {
            "actor_name": person_name,
            "embedding": mean_emb.tolist(),
            "model": f"insightface_{model_pack}",
            "num_images_used": used,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        out_path = outd / f"{slugify(person_name)}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)

        total_written += 1

    print("\nDONE")
    print(f"Written actor pkls: {total_written}")
    print(f"Skipped actors:     {total_skipped}")
    print(f"Output dir:         {outd}")


def main():
    ap = argparse.ArgumentParser("Create actor embeddings (InsightFace/ONNX GPU) -> one .pkl per actor")
    ap.add_argument("--dataset_folder", type=str, required=True, help="Root folder containing subfolders per actor")
    ap.add_argument("--output_dir", type=str, required=True, help="Where to write <actor>.pkl files")
    ap.add_argument("--model_pack", type=str, default="buffalo_l")
    ap.add_argument("--det_size", type=str, default="640,640", help="e.g. 640,640")
    ap.add_argument("--gpu_id", type=int, default=0, help="GPU id to use (set -1 for CPU)")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    w, h = [int(x.strip()) for x in args.det_size.split(",")]
    create_actor_pkls(
        dataset_folder=args.dataset_folder,
        output_dir=args.output_dir,
        model_pack=args.model_pack,
        det_size=(w, h),
        gpu_id=args.gpu_id,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
