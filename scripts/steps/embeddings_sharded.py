from __future__ import annotations
import core.deps as D
from core.utils import ensure_dir, touch, iter_ndjson

def run_embeddings_sharded(cfg, faces_dir: str, emb_dir: str, emb_pkl_dir: str, emb_done: str, gpu_cfg) -> None:
    ray = D.ray()
    torch = D.torch()
    cv2 = D.cv2()
    np = D.np()
    FaceAnalysis = D.FaceAnalysis()
    tqdm = D.tqdm()

    from concurrent.futures import ThreadPoolExecutor

    ensure_dir(emb_dir)
    if cfg.export_pkl:
        ensure_dir(emb_pkl_dir)

    if gpu_cfg.use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but gpu.use_cuda=true")
    if gpu_cfg.use_cuda and torch.cuda.device_count() <= 0:
        raise RuntimeError("No visible GPUs.")

    visible_gpu_count = torch.cuda.device_count() if gpu_cfg.use_cuda else 0
    num_actors = visible_gpu_count
    if gpu_cfg.max_actors and gpu_cfg.max_actors > 0:
        num_actors = min(num_actors, int(gpu_cfg.max_actors))
    if num_actors <= 0:
        raise RuntimeError("Embeddings require GPUs.")

    face_files = sorted([p for p in D.Path(faces_dir).glob("shot_*.ndjson")])
    if not face_files:
        raise RuntimeError(f"No face shards found in {faces_dir}. Run face_detection first.")

    @ray.remote(num_gpus=gpu_cfg.gpus_per_actor, num_cpus=int(cfg.num_cpus_per_actor))
    class GPUEmbedder:
        def __init__(self, model_pack: str, det_size, prefetch_threads: int, export_pkl: bool, pkl_protocol: int):
            self.model_pack = model_pack
            self.det_size = tuple(det_size)
            self.app = FaceAnalysis(name=model_pack, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            self.prefetch_threads = int(prefetch_threads)
            self.export_pkl = bool(export_pkl)
            self.pkl_protocol = int(pkl_protocol)

        def _read_decode(self, path: str):
            try:
                with open(path, "rb") as f:
                    b = f.read()
                arr = np.frombuffer(b, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return path, img
            except Exception:
                return path, None

        @staticmethod
        def _iou(a, b) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            return inter / (area_a + area_b - inter + 1e-9)

        def process_records(self, records, shot: str, emb_dir: str, emb_pkl_dir: str, faces_per_part: int):
            out_shot_dir = D.Path(emb_dir) / shot
            out_shot_dir.mkdir(parents=True, exist_ok=True)

            out_pkl_shot_dir = None
            if self.export_pkl:
                out_pkl_shot_dir = D.Path(emb_pkl_dir) / shot
                out_pkl_shot_dir.mkdir(parents=True, exist_ok=True)

            existing = sorted(out_shot_dir.glob("part_*.npy"))
            part_idx = len(existing)

            embs_buf = []
            meta_lines = []

            def flush_part():
                nonlocal part_idx, embs_buf, meta_lines
                if not embs_buf:
                    return

                E = np.vstack(embs_buf).astype(np.float32)  # (N,512)

                part_path = out_shot_dir / f"part_{part_idx:05d}.npy"
                meta_path = out_shot_dir / f"meta_{part_idx:05d}.ndjson"
                np.save(part_path, E)

                with open(meta_path, "w", encoding="utf-8") as f:
                    for m in meta_lines:
                        f.write(D.json.dumps(m, ensure_ascii=False) + "\n")

                # ✅ PKL shard aligned with NPY shard
                if self.export_pkl and out_pkl_shot_dir is not None:
                    pkl_path = out_pkl_shot_dir / f"part_{part_idx:05d}.pkl"
                    payload = {
                        "format": "embedding_shard_v1",
                        "model_pack": self.model_pack,
                        "det_size": self.det_size,
                        "embeddings": E,     # np.ndarray (N,512) float32
                        "meta": meta_lines,  # list[dict], same order as embeddings rows
                    }
                    with open(pkl_path, "wb") as pf:
                        D.pickle.dump(payload, pf, protocol=self.pkl_protocol)

                part_idx += 1
                embs_buf = []
                meta_lines = []

            paths = [r["image_path"] for r in records]
            with ThreadPoolExecutor(max_workers=self.prefetch_threads) as ex:
                decoded = list(ex.map(self._read_decode, paths))

            for rec, (p, img) in zip(records, decoded):
                if img is None:
                    continue

                faces_yolo = rec.get("faces", [])
                if not faces_yolo:
                    continue

                faces_if = self.app.get(img)
                if not faces_if:
                    continue

                if_boxes = [tuple(map(float, f.bbox.tolist())) for f in faces_if]

                for f in faces_yolo:
                    k = int(f.get("k", 0))
                    yb = (float(f["x1"]), float(f["y1"]), float(f["x2"]), float(f["y2"]))

                    best_i, best_iou = -1, -1.0
                    for i, bb in enumerate(if_boxes):
                        v = self._iou(yb, bb)
                        if v > best_iou:
                            best_iou, best_i = v, i
                    if best_i < 0:
                        continue

                    emb = faces_if[best_i].normed_embedding.astype(np.float32)
                    embs_buf.append(emb.reshape(1, -1))

                    meta_lines.append({
                        "face_uid": f"{p}#{k}",
                        "image_path": p,
                        "bbox": [int(yb[0]), int(yb[1]), int(yb[2]), int(yb[3])],
                        "confidence": float(f.get("confidence", 0.0)),
                    })

                    if len(meta_lines) >= faces_per_part:
                        flush_part()

            flush_part()
            return {"records": len(records)}

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_gpus=visible_gpu_count, ignore_reinit_error=True, include_dashboard=False)

    actors = [
        GPUEmbedder.remote(
            cfg.insightface_model_pack,
            cfg.det_size,
            cfg.prefetch_threads,
            cfg.export_pkl,
            cfg.pkl_protocol,
        )
        for _ in range(num_actors)
    ]

    futures = []
    pbar = tqdm(desc="Embeddings (images)", unit="img")
    processed_images = 0

    submit_counter = 0
    def submit(shot: str, records, idx: int):
        nonlocal submit_counter
        actor = actors[idx % len(actors)]
        submit_counter += 1
        futures.append(actor.process_records.remote(records, shot, emb_dir, emb_pkl_dir, cfg.faces_per_part))

    for si, face_file in enumerate(face_files):
        shot = face_file.stem
        batch = []

        for rec in iter_ndjson(str(face_file)):
            batch.append(rec)

            if cfg.max_images and cfg.max_images > 0 and processed_images >= cfg.max_images:
                break

            if len(batch) >= 256:
                submit(shot, batch, si)
                processed_images += len(batch)
                pbar.update(len(batch))
                batch = []

                if len(futures) >= cfg.max_in_flight_batches:
                    ready, futures[:] = ray.wait(futures, num_returns=min(cfg.drain_batch_count, len(futures)))
                    ray.get(ready)

        if batch:
            submit(shot, batch, si)
            processed_images += len(batch)
            pbar.update(len(batch))

        if cfg.max_images and cfg.max_images > 0 and processed_images >= cfg.max_images:
            break

    while futures:
        ready, futures[:] = ray.wait(futures, num_returns=min(cfg.drain_batch_count, len(futures)))
        ray.get(ready)

    pbar.close()
    ray.shutdown()

    touch(emb_done, f"done={D.time.time()}\n")
    print(f"[Embed] DONE -> {emb_done}")
    print(f"[Embed] NPY shards -> {emb_dir}")
    if cfg.export_pkl:
        print(f"[Embed] PKL shards -> {emb_pkl_dir}")
