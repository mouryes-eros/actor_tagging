from __future__ import annotations
import core.deps as D
from core.utils import iter_ndjson

def draw_shot(ndjson_path: str, out_dir: str, threshold: float, box_width: int, font_size: int):
    Image = D.PIL_Image()
    ImageDraw = D.PIL_ImageDraw()
    ImageFont = D.PIL_ImageFont()

    out_dir = D.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for rec in iter_ndjson(ndjson_path):
        img_path = rec["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        d = ImageDraw.Draw(img)
        for det in rec.get("detections", []):
            if threshold > 0 and float(det["similarity"]) < threshold:
                continue
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['display_name']} ({float(det['similarity']):.3f})"
            d.rectangle((x1, y1, x2, y2), outline="yellow", width=box_width)
            d.text((x1, max(5, y1 - 20)), label, fill="white", font=font)

        img.save(out_dir / D.Path(img_path).name)

def main():
    ap = D.argparse.ArgumentParser()
    ap.add_argument("--movie", type=str, required=True)
    ap.add_argument("--shot", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=0.0)
    args = ap.parse_args()

    ndjson_path = str(D.Path("outputs") / args.movie / "metadata" / f"{args.shot}.ndjson")
    out_dir = str(D.Path("visualize/output") / args.movie / args.shot)

    draw_shot(ndjson_path, out_dir, float(args.threshold), box_width=3, font_size=18)
    print("[Viz] done:", out_dir)

if __name__ == "__main__":
    main()
