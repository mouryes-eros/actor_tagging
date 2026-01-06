from __future__ import annotations
import core.deps as D
from core.utils import ensure_dir, touch, iter_ndjson

def run_actor_assign_matmul(cfg_assign, emb_dir: str, assign_dir: str, assign_done: str,
                            actor_dir: str, actor_keys, actor_embedding_files: dict, actor_display_names: dict):
    np = D.np()
    ensure_dir(assign_dir)

    def load_actor_matrix():
        keys, embs = [], []
        for k in actor_keys:
            fn = actor_embedding_files.get(k, f"{k}.pkl")
            p = D.Path(actor_dir) / fn
            if not p.exists():
                print("[Assign] missing actor embedding:", p)
                continue
            with open(p, "rb") as f:
                d = D.pickle.load(f)
            e = np.array(d["embedding"], dtype=np.float32).reshape(1, -1)[0]
            embs.append(e)
            keys.append(k)
        if not embs:
            raise RuntimeError("[Assign] No actor embeddings found for this movie.")
        A = np.vstack(embs).astype(np.float32)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        return keys, A

    used_keys, A = load_actor_matrix()

    shot_dirs = sorted([p for p in D.Path(emb_dir).glob("shot_*") if p.is_dir()])
    if not shot_dirs:
        raise RuntimeError(f"No shot dirs in {emb_dir}. Run embeddings first.")

    batch = 50000
    max_faces = int(cfg_assign.max_faces) if cfg_assign.max_faces else 0
    seen = 0

    for sd in shot_dirs:
        shot = sd.name
        out_shot_dir = D.Path(assign_dir) / shot
        out_shot_dir.mkdir(parents=True, exist_ok=True)

        parts = sorted(sd.glob("part_*.npy"))
        for part_path in parts:
            idx = part_path.stem.split("_")[-1]
            meta_path = sd / f"meta_{idx}.ndjson"
            if not meta_path.exists():
                raise RuntimeError(f"Missing meta for {part_path}: {meta_path}")

            out_path = out_shot_dir / f"part_{idx}.ndjson"
            if out_path.exists():
                continue

            E = np.load(part_path, mmap_mode="r")
            N = E.shape[0]
            meta = list(iter_ndjson(str(meta_path)))
            if len(meta) != N:
                raise RuntimeError(f"Meta/E mismatch: {meta_path} has {len(meta)} lines but {part_path} has {N} rows")

            with open(out_path, "w", encoding="utf-8") as f:
                for i0 in range(0, N, batch):
                    Eb = np.asarray(E[i0:i0+batch], dtype=np.float32)
                    Eb = Eb / (np.linalg.norm(Eb, axis=1, keepdims=True) + 1e-8)

                    S = Eb @ A.T
                    best = np.argmax(S, axis=1)
                    sim = S[np.arange(S.shape[0]), best]

                    for j in range(S.shape[0]):
                        if max_faces and seen >= max_faces:
                            break
                        m = meta[i0 + j]
                        actor_key = used_keys[int(best[j])]
                        score = float(sim[j])

                        actor_out = actor_key if (cfg_assign.similarity_threshold <= 0 or score >= cfg_assign.similarity_threshold) else cfg_assign.unknown_label

                        rec = {
                            "face_uid": m["face_uid"],
                            "image_path": m["image_path"],
                            "bbox": m["bbox"],
                            "confidence": m.get("confidence", None),
                            "actor": actor_out,
                            "display_name": actor_display_names.get(actor_key, actor_key.replace("_", " ").title()),
                            "similarity": score,
                        }
                        f.write(D.json.dumps(rec, ensure_ascii=False) + "\n")
                        seen += 1

                    if max_faces and seen >= max_faces:
                        break

            print(f"[Assign MatMul] wrote {out_path}")
            if max_faces and seen >= max_faces:
                break

        if max_faces and seen >= max_faces:
            break

    touch(assign_done, f"done={D.time.time()}\n")
    print(f"[Assign MatMul] DONE -> {assign_done}")
