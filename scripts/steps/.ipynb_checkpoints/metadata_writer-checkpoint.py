from __future__ import annotations
import core.deps as D
from core.utils import ensure_dir, touch, iter_ndjson

def write_metadata(cfg, movie_name: str, assign_dir: str, metadata_dir: str, metadata_done: str, unknown_label: str):
    ensure_dir(metadata_dir)

    shot_dirs = sorted([p for p in D.Path(assign_dir).glob("shot_*") if p.is_dir()])
    if not shot_dirs:
        raise RuntimeError(f"No assignments in {assign_dir}. Run actor_assign first.")

    for sd in shot_dirs:
        shot = sd.name
        out_path = D.Path(metadata_dir) / f"{shot}.ndjson"
        if out_path.exists():
            continue

        current_img = None
        dets = []

        def flush(img_path, dets_list):
            if img_path is None:
                return
            rec = {"movie": movie_name, "image_path": img_path, "detections": dets_list}
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(D.json.dumps(rec, ensure_ascii=False) + "\n")

        parts = sorted(sd.glob("part_*.ndjson"))
        for part in parts:
            for r in iter_ndjson(str(part)):
                if cfg.similarity_threshold > 0 and float(r["similarity"]) < cfg.similarity_threshold:
                    continue
                if not cfg.keep_unknown and r["actor"] == unknown_label:
                    continue

                img = r["image_path"]
                if current_img is None:
                    current_img = img

                if img != current_img:
                    flush(current_img, dets)
                    current_img = img
                    dets = []

                dets.append({
                    "face_uid": r["face_uid"],
                    "actor": r["actor"],
                    "display_name": r.get("display_name", r["actor"]),
                    "similarity": float(r["similarity"]),
                    "bbox": r["bbox"],
                    "confidence": r.get("confidence", None),
                })

        flush(current_img, dets)
        print(f"[Metadata] wrote {out_path}")

    touch(metadata_done, f"done={D.time.time()}\n")
    print(f"[Metadata] DONE -> {metadata_done}")
