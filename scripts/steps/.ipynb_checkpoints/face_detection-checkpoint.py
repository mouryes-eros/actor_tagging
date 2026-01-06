from __future__ import annotations
import core.deps as D
from core.utils import ensure_dir, touch, shot_from_path, ndjson_append, walk_images

def run_face_detection(cfg, movie_frames_dir: str, faces_dir: str, faces_done: str, gpu_cfg) -> None:
    ray = D.ray()
    torch = D.torch()
    YOLO = D.YOLO()
    np = D.np()
    cv2 = D.cv2()
    tqdm = D.tqdm()

    from concurrent.futures import ThreadPoolExecutor

    ensure_dir(faces_dir)

    if gpu_cfg.use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but gpu.use_cuda=true")
    if gpu_cfg.use_cuda and torch.cuda.device_count() <= 0:
        raise RuntimeError("No visible GPUs.")

    D.os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    D.os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    D.os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    visible_gpu_count = torch.cuda.device_count() if gpu_cfg.use_cuda else 0
    num_actors = visible_gpu_count
    if gpu_cfg.max_actors and gpu_cfg.max_actors > 0:
        num_actors = min(num_actors, int(gpu_cfg.max_actors))
    if num_actors <= 0:
        raise RuntimeError("Face detection requires GPUs.")

    @ray.remote(num_gpus=gpu_cfg.gpus_per_actor, num_cpus=int(cfg.num_cpus_per_actor))
    class GPUFaceDetector:
        def __init__(self, model_path: str, imgsz: int, conf: float, prefetch_threads: int):
            self.model = YOLO(model_path)
            self.model.to("cuda:0")
            try:
                self.model.fuse()
            except Exception:
                pass
            self.imgsz = int(imgsz)
            self.conf = float(conf)
            self.prefetch_threads = int(prefetch_threads)

        def _read_decode(self, path: str):
            try:
                with open(path, "rb") as f:
                    b = f.read()
                arr = np.frombuffer(b, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return path, img
            except Exception:
                return path, None

        def detect_batch(self, image_paths):
            imgs, ok_paths = [], []
            with ThreadPoolExecutor(max_workers=self.prefetch_threads) as ex:
                for p, img in ex.map(self._read_decode, image_paths):
                    if img is not None:
                        ok_paths.append(p)
                        imgs.append(img)

            if not imgs:
                return {"processed": len(image_paths), "records": []}

            results = self.model.predict(
                source=imgs,
                verbose=False,
                device="cuda:0",
                imgsz=self.imgsz,
                conf=self.conf,
                workers=0,
                half=True,
            )

            records = []
            for r, original_path in zip(results, ok_paths):
                faces = []
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.detach().cpu().numpy()
                    confs = r.boxes.conf.detach().cpu().numpy()
                    for j in range(xyxy.shape[0]):
                        b = xyxy[j]
                        faces.append({
                            "x1": float(b[0]), "y1": float(b[1]),
                            "x2": float(b[2]), "y2": float(b[3]),
                            "confidence": float(confs[j]),
                            "k": int(j),
                        })
                records.append({"image_path": original_path, "faces": faces})

            return {"processed": len(image_paths), "records": records}

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_gpus=visible_gpu_count, ignore_reinit_error=True, include_dashboard=False)

    actors = [GPUFaceDetector.remote(cfg.yolo_model_path, cfg.imgsz, cfg.conf, cfg.prefetch_threads) for _ in range(num_actors)]

    def chunked(it, n):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) >= n:
                yield buf
                buf = []
        if buf:
            yield buf

    futures = []
    pbar = tqdm(desc="Face detection (completed)", unit="img")
    completed = 0

    def flush(res_list):
        nonlocal completed
        for res in res_list:
            completed += int(res["processed"])
            for rec in res["records"]:
                shot = shot_from_path(rec["image_path"])
                out_path = str(D.Path(faces_dir) / f"{shot}.ndjson")
                ndjson_append(out_path, rec)
        pbar.n = completed
        pbar.refresh()

    img_iter = walk_images(movie_frames_dir, cfg.valid_exts)

    for bi, batch in enumerate(chunked(img_iter, cfg.batch_size_images)):
        actor = actors[bi % len(actors)]
        futures.append(actor.detect_batch.remote(batch))

        if len(futures) >= cfg.max_in_flight_batches:
            ready, futures = ray.wait(futures, num_returns=min(cfg.drain_batch_count, len(futures)))
            flush(ray.get(ready))

    while futures:
        ready, futures = ray.wait(futures, num_returns=min(cfg.drain_batch_count, len(futures)))
        flush(ray.get(ready))

    pbar.close()
    ray.shutdown()

    touch(faces_done, f"done={D.time.time()}\n")
    print(f"[FaceDet] DONE -> {faces_done}")
