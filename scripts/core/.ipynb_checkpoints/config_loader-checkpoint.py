from __future__ import annotations
import core.deps as D

from core.types import (
    MovieConfig, GPUConfig, PathsConfig,
    FaceDetectionConfig, EmbeddingConfig,
    ActorAssignConfig, MetadataConfig,
    ResumeConfig, PipelineConfig
)

def _tup(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)

def _fmt(v: str, movie: str) -> str:
    return v.replace("{movie}", movie)

def _repo_root() -> D.Path:
    return D.Path(__file__).resolve().parents[1]

def load_raw_toml(path: str | None = None) -> dict:
    toml = D.toml()
    cfg_path = D.Path(path) if path else (_repo_root() / "config" / "pipeline.toml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return toml.loads(cfg_path.read_text(encoding="utf-8"))

def load_config(path: str | None = None, override_movie: str | None = None) -> PipelineConfig:
    data = load_raw_toml(path)

    movies = data.get("movies") or {}
    if not movies:
        raise KeyError("No [movies.*] blocks found in pipeline.toml")

    movie_section = data.get("movie") or {}
    active_from_toml = movie_section.get("active", None)

    if override_movie:
        active_movie = override_movie
    elif active_from_toml:
        active_movie = active_from_toml
    else:
        active_movie = next(iter(movies.keys()))

    if active_movie not in movies:
        raise KeyError(f"movie '{active_movie}' not found under [movies.*]. Available: {list(movies.keys())}")

    mb = movies[active_movie]
    movie = MovieConfig(
        active=active_movie,
        frames_subdir=str(mb["frames_subdir"]),
        actors=list(mb["actors"]),
        actor_embedding_files=dict(mb.get("actor_embedding_files", {})),
        actor_display_names=dict(mb.get("actor_display_names", {})),
    )

    p = data["paths"]
    paths = PathsConfig(
        frames_root=str(p["frames_root"]),
        actor_dir=str(p["actor_dir"]),
        base_out=_fmt(str(p["base_out"]), movie.active),

        faces_dir=_fmt(str(p["faces_dir"]), movie.active),
        faces_done=_fmt(str(p["faces_done"]), movie.active),

        emb_dir=_fmt(str(p["emb_dir"]), movie.active),
        emb_done=_fmt(str(p["emb_done"]), movie.active),

        emb_pkl_dir=_fmt(str(p["emb_pkl_dir"]), movie.active),

        assign_dir=_fmt(str(p["assign_dir"]), movie.active),
        assign_done=_fmt(str(p["assign_done"]), movie.active),

        metadata_dir=_fmt(str(p["metadata_dir"]), movie.active),
        metadata_done=_fmt(str(p["metadata_done"]), movie.active),

        viz_out_dir=_fmt(str(p["viz_out_dir"]), movie.active),
    )

    gpu = GPUConfig(**data["gpu"])

    fd = data["face_detection"]
    face_detection = FaceDetectionConfig(
        enabled=bool(fd["enabled"]),
        yolo_model_path=str(fd["yolo_model_path"]),
        imgsz=int(fd["imgsz"]),
        conf=float(fd["conf"]),
        valid_exts=_tup(fd["valid_exts"]),
        batch_size_images=int(fd["batch_size_images"]),
        max_in_flight_batches=int(fd["max_in_flight_batches"]),
        drain_batch_count=int(fd["drain_batch_count"]),
        num_cpus_per_actor=int(fd["num_cpus_per_actor"]),
        prefetch_threads=int(fd["prefetch_threads"]),
    )

    em = data["embedding"]
    embedding = EmbeddingConfig(
        enabled=bool(em["enabled"]),
        insightface_model_pack=str(em["insightface_model_pack"]),
        det_size=(int(em["det_size"][0]), int(em["det_size"][1])),
        faces_per_part=int(em["faces_per_part"]),
        max_images=int(em["max_images"]),
        num_cpus_per_actor=int(em["num_cpus_per_actor"]),
        prefetch_threads=int(em["prefetch_threads"]),
        max_in_flight_batches=int(em["max_in_flight_batches"]),
        drain_batch_count=int(em["drain_batch_count"]),
        export_pkl=bool(em.get("export_pkl", True)),
        pkl_protocol=int(em.get("pkl_protocol", 4)),
    )

    aa = data["actor_assign"]
    actor_assign = ActorAssignConfig(
        enabled=bool(aa["enabled"]),
        unknown_label=str(aa["unknown_label"]),
        max_faces=int(aa["max_faces"]),
        similarity_threshold=float(aa["similarity_threshold"]),
    )

    md = data["metadata"]
    metadata = MetadataConfig(
        enabled=bool(md["enabled"]),
        similarity_threshold=float(md["similarity_threshold"]),
        keep_unknown=bool(md["keep_unknown"]),
    )

    resume = ResumeConfig(**data["resume"])

    return PipelineConfig(
        movie=movie,
        gpu=gpu,
        paths=paths,
        face_detection=face_detection,
        embedding=embedding,
        actor_assign=actor_assign,
        metadata=metadata,
        resume=resume,
    )
