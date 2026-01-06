from __future__ import annotations
import core.deps as D

def done_marker_exists(path: str) -> bool:
    return D.Path(path).exists()

def should_skip(step: str, cfg) -> bool:
    if not cfg.resume.enabled:
        return False

    if step == "face_detection" and cfg.resume.face_detection:
        return done_marker_exists(cfg.paths.faces_done)

    if step == "embedding_generation" and cfg.resume.embedding_generation:
        return done_marker_exists(cfg.paths.emb_done)

    if step == "actor_assign" and cfg.resume.actor_assign:
        return done_marker_exists(cfg.paths.assign_done)

    if step == "metadata_generation" and cfg.resume.metadata_generation:
        return done_marker_exists(cfg.paths.metadata_done)

    return False
