from __future__ import annotations
import core.deps as D

def ensure_dir(path: str) -> None:
    D.Path(path).mkdir(parents=True, exist_ok=True)

def touch(path: str, text: str = "") -> None:
    p = D.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def shot_from_path(image_path: str) -> str:
    for part in D.Path(image_path).parts:
        if part.startswith("shot_"):
            return part
    return "shot_all"

def ndjson_append(path: str, obj: dict) -> None:
    p = D.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(D.json.dumps(obj, ensure_ascii=False) + "\n")

def iter_ndjson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield D.json.loads(line)

def walk_images(root: str, valid_exts: tuple[str, ...]):
    ve = tuple(e.lower() for e in valid_exts)
    for dp, _, files in D.os.walk(root):
        for fn in files:
            if fn.lower().endswith(ve):
                yield str(D.Path(dp) / fn)
