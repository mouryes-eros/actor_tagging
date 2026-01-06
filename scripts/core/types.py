from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class MovieConfig:
    active: str
    frames_subdir: str
    actors: List[str]
    actor_embedding_files: Dict[str, str]
    actor_display_names: Dict[str, str]

@dataclass
class GPUConfig:
    use_cuda: bool
    device_ids: List[int]
    gpus_per_actor: int
    max_actors: int

@dataclass
class PathsConfig:
    frames_root: str
    actor_dir: str
    base_out: str

    faces_dir: str
    faces_done: str

    emb_dir: str
    emb_done: str

    emb_pkl_dir: str

    assign_dir: str
    assign_done: str

    metadata_dir: str
    metadata_done: str

    viz_out_dir: str

@dataclass
class FaceDetectionConfig:
    enabled: bool
    yolo_model_path: str
    imgsz: int
    conf: float
    valid_exts: Tuple[str, ...]

    batch_size_images: int
    max_in_flight_batches: int
    drain_batch_count: int

    num_cpus_per_actor: int
    prefetch_threads: int

@dataclass
class EmbeddingConfig:
    enabled: bool
    insightface_model_pack: str
    det_size: Tuple[int, int]

    faces_per_part: int
    max_images: int

    num_cpus_per_actor: int
    prefetch_threads: int
    max_in_flight_batches: int
    drain_batch_count: int

    export_pkl: bool
    pkl_protocol: int

@dataclass
class ActorAssignConfig:
    enabled: bool
    unknown_label: str
    max_faces: int
    similarity_threshold: float

@dataclass
class MetadataConfig:
    enabled: bool
    similarity_threshold: float
    keep_unknown: bool

@dataclass
class ResumeConfig:
    enabled: bool
    face_detection: bool
    embedding_generation: bool
    actor_assign: bool
    metadata_generation: bool

@dataclass
class PipelineConfig:
    movie: MovieConfig
    gpu: GPUConfig
    paths: PathsConfig
    face_detection: FaceDetectionConfig
    embedding: EmbeddingConfig
    actor_assign: ActorAssignConfig
    metadata: MetadataConfig
    resume: ResumeConfig
