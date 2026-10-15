#!/usr/bin/env python3
"""
Person-emotion annotation tool (local, desktop browser).

    pip install flask
    python app.py                 # HECO, then open http://127.0.0.1:8000
    python app.py --dataset caer  # CAER-S multi-person frames (run make_caer_multi.py first)

Reads the metadata JSON and annotation_schema.json. The form is built
from the schema (image_questions once per image, person_questions per person).
Every answer is written to the output JSON (annotations.json for HECO,
annotations_caer.json for CAER) on "Next".

Boxes are stored in original image pixels [x1, y1, x2, y2], the same space as
body_bbox in the metadata, so they don't depend on the window/display size.

If annotations.json is still in the old (v1) format it is renamed to
annotations_v1.json and used only to pre-fill emotion and clothing.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

HERE = Path(__file__).resolve().parent
# dataset category_7 label -> schema emotion value
EMOTION_MAP = {"Happy": "happy", "Sad": "sad", "Anger": "angry", "Angry": "angry", "Fear": "fearful",
               "Surprise": "surprised", "Disgust": "disgusted", "Neutral": "neutral"}

app = Flask(__name__, static_folder=None)
lock = threading.Lock()

SCHEMA: dict = {}
IMAGES: list[Path] = []            # image index -> file on disk
IMAGE_NAMES: list[str] = []        # image index -> filename (unique key)
IMAGE_META: list[dict] = []        # image index -> {"image_id", optional "scene_dataset"}
IMAGE_ITEMS: list[list[dict]] = [] # image index -> its people (entries of ITEMS)
ITEMS: list[dict] = []             # one entry per person, in annotation order
SCENES: dict[str, dict] = {}       # filename -> {"scene_mood": ..., "image_size": [W, H], ...}
PEOPLE: dict[str, dict] = {}       # "filename::person" -> saved person record
LEGACY: dict[str, dict] = {}       # "filename::person" -> v1 record (pre-fill only)
CFG: dict = {}


def key(filename: str, person: int) -> str:
    return f"{filename}::{person}"


def resolve_image(img: dict, fallback_dir: Path) -> Path | None:
    """Use image_path from the JSON; if it isn't on this machine, look for the filename in fallback_dir."""
    for p in (Path(img["image_path"]), fallback_dir / img["filename"]):
        if p.is_file():
            return p
    return None


def load_metadata(path: Path, fallback_dir: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for img in data:
        f = resolve_image(img, fallback_dir)
        if f is None:
            print(f"  ! missing image, skipped: {img['image_path']}")
            continue
        people = img.get("people", [])
        if not people:
            continue
        IMAGES.append(f)
        IMAGE_NAMES.append(img["filename"])
        meta = {"image_id": img.get("image_id") or Path(img["filename"]).stem}
        if EMOTION_MAP.get(img.get("image_label")):   # image-level label, e.g. CAER folder
            meta["scene_dataset"] = EMOTION_MAP[img["image_label"]]
        IMAGE_META.append(meta)
        IMAGE_ITEMS.append([])
        img_idx = len(IMAGES) - 1
        for n, p in enumerate(people):
            ITEMS.append({
                "image": img_idx,
                "filename": img["filename"],
                "person": n,
                "people_in_image": len(people),
                "bbox": p["body_bbox"],
                "emotion": EMOTION_MAP.get(p.get("category_7")),
                "description": p.get("cloth_description", "") or "",
            })
            IMAGE_ITEMS[img_idx].append(ITEMS[-1])


# ---- Schema logic (mirrored in index.html) ---------------------------------
def visible_ids(questions: list[dict], answers: dict) -> set[str]:
    """A question is shown when every show_if condition holds. A condition on a
    hidden question fails; on an unanswered one, `not_in` passes and `in` fails."""
    shown: set[str] = set()
    for q in questions:  # show_if only refers to earlier questions
        ok = True
        for ref, cond in (q.get("show_if") or {}).items():
            if ref not in shown:
                ok = False
            elif ref not in answers:
                ok = "in" not in cond
            elif "in" in cond:
                ok = answers[ref] in cond["in"]
            else:
                ok = answers[ref] not in cond.get("not_in", [])
            if not ok:
                break
        if ok:
            shown.add(q["id"])
    return shown


def clean_answers(questions: list[dict], raw: dict) -> tuple[dict, list[str]]:
    """Keep valid answers to visible questions; return (answers, missing labels)."""
    raw = raw if isinstance(raw, dict) else {}
    answers: dict = {}
    for q in questions:
        v = raw.get(q["id"])
        values = [o["value"] for o in q.get("options", [])]
        ok = False
        if q["type"] == "single_choice":
            # compare type too, so True never passes for 1
            ok = any(v == x and type(v) is type(x) for x in values)
        elif q["type"] == "multi_choice":
            ok = isinstance(v, list) and all(x in values for x in v)
            if ok:
                v = [x for x in values if x in v]  # schema order, no duplicates
        elif q["type"] == "text":
            ok = isinstance(v, str)
            if ok:
                v = v.strip()
        if ok:
            answers[q["id"]] = v
    shown = visible_ids(questions, answers)
    answers = {k: v for k, v in answers.items() if k in shown}
    missing = [q["question"] for q in questions
               if q["id"] in shown and q.get("required")
               and answers.get(q["id"]) in (None, "", [])]
    return answers, missing


# ---- Storage ---------------------------------------------------------------
def load_annotations(out: Path) -> None:
    legacy_path = out.with_name(out.stem + "_v1.json")
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        if "images" not in data and "annotations" in data:  # old format
            if legacy_path.exists():
                raise SystemExit(f"{out} is in the old format but {legacy_path} already exists; "
                                 "move one of them away and restart.")
            os.replace(out, legacy_path)
            print(f"  Old-format annotations moved to {legacy_path} (used to pre-fill)")
        else:
            for img in data.get("images", []):
                fn = img["filename"]
                SCENES[fn] = {k: v for k, v in img.items() if k not in ("people", "image_id", "filename", "scene_dataset")}
                for p in img.get("people", []):
                    PEOPLE[key(fn, p["person_id"])] = p
    if legacy_path.exists():
        for rec in json.loads(legacy_path.read_text(encoding="utf-8")).get("annotations", []):
            LEGACY[key(rec["filename"], rec["person"])] = rec


def write_json(path: Path, obj) -> None:
    """Atomic write: a crash can never leave a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_annotations() -> None:
    images = []
    for idx, fn in enumerate(IMAGE_NAMES):
        people = [PEOPLE[k] for k in (key(fn, it["person"]) for it in IMAGE_ITEMS[idx]) if k in PEOPLE]
        if not people and fn not in SCENES:
            continue
        images.append({**IMAGE_META[idx], "filename": fn, **SCENES.get(fn, {}), "people": people})
    write_json(CFG["out"], {"schema_name": SCHEMA.get("schema_name"), "schema_version": SCHEMA.get("version"),
                            "total_people": len(ITEMS), "annotated_people": done_count(),
                            "box_format": "xyxy, original image pixels",
                            "images": images})


def done_count() -> int:
    return sum(1 for it in ITEMS if key(it["filename"], it["person"]) in PEOPLE)



def current_box(it: dict) -> list[float]:
    rec = PEOPLE.get(key(it["filename"], it["person"]))
    return rec["box"] if rec else it["bbox"]


def prefill(it: dict) -> dict:
    """Starting answers for a person who hasn't been saved in the new format yet."""
    a = {"clothing": it["description"]}
    old = LEGACY.get(key(it["filename"], it["person"]))
    if old:
        a["clothing"] = old.get("description", it["description"])
        if old.get("emotion") in EMOTION_MAP:
            a["emotion"] = EMOTION_MAP[old["emotion"]]
    return a


# ---- Routes ----------------------------------------------------------------
@app.get("/")
def index():
    return send_file(HERE / "static" / "index.html")


@app.get("/api/config")
def config():
    # continue after the furthest saved person, so skipped people (never saved) aren't shown again
    resume = max((i + 1 for i, it in enumerate(ITEMS) if key(it["filename"], it["person"]) in PEOPLE), default=0)
    return jsonify(schema=SCHEMA, total=len(ITEMS), done=done_count(), resume=resume)


@app.get("/api/item/<int:i>")
def item(i: int):
    if not 0 <= i < len(ITEMS):
        abort(404)
    it = ITEMS[i]
    rec = PEOPLE.get(key(it["filename"], it["person"]))
    others = [current_box(o) for o in IMAGE_ITEMS[it["image"]] if o["person"] != it["person"]]
    nxt = ITEMS[i + 1]["image"] if i + 1 < len(ITEMS) else None
    return jsonify(
        index=i, total=len(ITEMS), done=done_count(),
        item={**it, "other_boxes": others, "scene_dataset": IMAGE_META[it["image"]].get("scene_dataset")},
        image_url=f"/image/{it['image']}",
        next_image_url=f"/image/{nxt}" if nxt is not None else None,
        scene=SCENES.get(it["filename"], {}),
        annotation=rec,
        prefill=None if rec else prefill(it),
        legacy=key(it["filename"], it["person"]) in LEGACY,
    )


@app.get("/image/<int:n>")
def image(n: int):
    # Only images listed in the metadata can be served, looked up by index.
    if not 0 <= n < len(IMAGES):
        abort(404)
    return send_file(IMAGES[n], max_age=3600)


def numbers(v, n: int) -> bool:
    return isinstance(v, list) and len(v) == n and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and x == x for x in v)


def clean_box(box, size) -> list[float] | None:
    """Clamp an [x1, y1, x2, y2] box (image pixels) to the image; None if invalid."""
    if not (numbers(box, 4) and numbers(size, 2)):
        return None
    W, H = size
    x1, y1, x2, y2 = box
    x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W)))
    y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


@app.post("/api/annotate")
def annotate():
    d = request.get_json(silent=True) or {}
    i = d.get("index")
    if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(ITEMS):
        return jsonify(error="invalid index"), 400
    it = ITEMS[i]

    scene, miss_scene = clean_answers(SCHEMA["image_questions"], d.get("scene"))
    person, miss_person = clean_answers(SCHEMA["person_questions"], d.get("person"))
    if miss_scene or miss_person:
        return jsonify(error="please answer: " + "; ".join(miss_scene + miss_person)), 400

    size = d.get("image_size")  # naturalWidth/Height of the image the browser drew the box on
    if not (numbers(size, 2) and min(size) > 0):
        return jsonify(error="invalid image size"), 400
    if person["box_check"] == "adjust":
        box = clean_box(d.get("box"), size)
        if box is None:
            return jsonify(error="invalid box"), 400
    else:
        box = it["bbox"]

    rec = {
        "person_id": it["person"],
        "box": box,
        "box_original": it["bbox"],
        **person,
        "emotion_dataset": it["emotion"],
        "clothing_original": it["description"],
        **({"clothing_edited": person["clothing"] != it["description"].strip()} if "clothing" in person else {}),
        "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with lock:
        SCENES[it["filename"]] = {**scene, "image_size": size}
        PEOPLE[key(it["filename"], it["person"])] = rec
        save_annotations()
    return jsonify(ok=True, next=i + 1, done=done_count())


# --dataset presets: (metadata, fallback image folder, output); any of them can be overridden
DATASETS = {
    "heco": (HERE / "heco_metadata_7emo.json", HERE / "Data", HERE / "annotations.json"),
    "caer": (HERE / "caer_multi_person.json", HERE / "CAER-S" / "CAER-S", HERE / "annotations_caer.json"),
    "meld": (HERE / "MELD_test_clothes.json", HERE / "MELD_Frames", HERE / "annotations_meld.json"),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Local person-emotion annotation tool")
    p.add_argument("--dataset", choices=DATASETS, default="heco")
    p.add_argument("--metadata", type=Path)
    p.add_argument("--schema", type=Path, default=HERE / "annotation_schema.json")
    p.add_argument("--images", type=Path,
                   help="fallback folder, used when image_path from the JSON doesn't exist")
    p.add_argument("--out", type=Path)
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    meta, images, out = DATASETS[a.dataset]
    a.metadata, a.images, a.out = a.metadata or meta, a.images or images, a.out or out
    if not a.metadata.exists():
        raise SystemExit(f"{a.metadata} not found" + (" - run make_caer_multi.py first" if a.dataset == "caer" else ""))

    SCHEMA.update(json.loads(a.schema.read_text(encoding="utf-8")))
    CFG["out"] = a.out.resolve()
    load_metadata(a.metadata, a.images.resolve())
    load_annotations(CFG["out"])

    print(f"\n  {len(ITEMS)} people in {len(IMAGES)} images, {done_count()} already annotated")
    if LEGACY:
        print(f"  {len(LEGACY)} old-format answers available as pre-fill")
    print(f"  Metadata {a.metadata}")
    print(f"  Saving to {CFG['out']}")
    print(f"  Open http://127.0.0.1:{a.port}\n")
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
