from __future__ import annotations
import core.deps as D

from core.config_loader import load_config, load_raw_toml
from core.resume import should_skip

from steps.face_detection import run_face_detection
from steps.embeddings_sharded import run_embeddings_sharded
from steps.actor_assign_matmul import run_actor_assign_matmul
from steps.metadata_writer import write_metadata

def _apply_gpu_env(cfg):
    if cfg.gpu.use_cuda and cfg.gpu.device_ids:
        D.os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in cfg.gpu.device_ids)

def run_one_movie(cfg):
    _apply_gpu_env(cfg)

    movie = cfg.movie.active
    movie_frames_dir = str(D.Path(cfg.paths.frames_root) / cfg.movie.frames_subdir)

    print("\n" + "=" * 90)
    print("RUN MOVIE:", movie)
    print("Frames:", movie_frames_dir)
    print("Out:", cfg.paths.base_out)
    print("=" * 90)

    if cfg.face_detection.enabled:
        if should_skip("face_detection", cfg):
            print("[Resume] skip face_detection")
        else:
            run_face_detection(cfg.face_detection, movie_frames_dir, cfg.paths.faces_dir, cfg.paths.faces_done, cfg.gpu)

    if cfg.embedding.enabled:
        if should_skip("embedding_generation", cfg):
            print("[Resume] skip embedding_generation")
        else:
            run_embeddings_sharded(
                cfg.embedding,
                faces_dir=cfg.paths.faces_dir,
                emb_dir=cfg.paths.emb_dir,
                emb_pkl_dir=cfg.paths.emb_pkl_dir,
                emb_done=cfg.paths.emb_done,
                gpu_cfg=cfg.gpu,
            )

    if cfg.actor_assign.enabled:
        if should_skip("actor_assign", cfg):
            print("[Resume] skip actor_assign")
        else:
            run_actor_assign_matmul(
                cfg_assign=cfg.actor_assign,
                emb_dir=cfg.paths.emb_dir,
                assign_dir=cfg.paths.assign_dir,
                assign_done=cfg.paths.assign_done,
                actor_dir=cfg.paths.actor_dir,
                actor_keys=cfg.movie.actors,
                actor_embedding_files=cfg.movie.actor_embedding_files,
                actor_display_names=cfg.movie.actor_display_names,
            )

    if cfg.metadata.enabled:
        if should_skip("metadata_generation", cfg):
            print("[Resume] skip metadata_generation")
        else:
            write_metadata(
                cfg=cfg.metadata,
                movie_name=movie,
                assign_dir=cfg.paths.assign_dir,
                metadata_dir=cfg.paths.metadata_dir,
                metadata_done=cfg.paths.metadata_done,
                unknown_label=cfg.actor_assign.unknown_label,
            )

def main():
    raw = load_raw_toml()
    mode = (raw.get("movie") or {}).get("mode", "single")
    movies = list((raw.get("movies") or {}).keys())

    if mode == "all":
        for m in movies:
            cfg = load_config(override_movie=m)
            run_one_movie(cfg)
    else:
        cfg = load_config()
        run_one_movie(cfg)

if __name__ == "__main__":
    main()
