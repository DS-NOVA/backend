import json
from pathlib import Path
from app.utils.frame_labels import CV_TEXT_MAP, PRED_TEXT_MAP

def export_frame_labels_json(video_dir: Path, graph_points: list[dict], cv_label_order: list[str]) -> Path:

    out_records = build_label_records(graph_points, cv_label_order)
    
    # JSON 저장 (legend 포함)
    json_path = video_dir / "frame_labels.json"
    json_obj = {
        "legend": {
            "cv": CV_TEXT_MAP,
            "pred": PRED_TEXT_MAP
        },
        "records": out_records
    }
    json_path.write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return json_path

def build_label_records(graph_points: list[dict], cv_label_order: list[str]) -> list[dict]:
    names = list(cv_label_order) + ["flash", "pattern", "redlight"]

    out_records = []
    for gp in graph_points:
        vec = [int(x) for x in (gp.get("labels") or [])]
        if not vec or sum(vec) == 0:
            continue

        label_kv = {name: (vec[i] if i < len(vec) else 0) for i, name in enumerate(names)}

        cv_texts = []
        for nm in cv_label_order:
            idx = names.index(nm)
            if idx < len(vec) and vec[idx] == 1 and nm in CV_TEXT_MAP:
                cv_texts.append(CV_TEXT_MAP[nm])

        pred_texts = []
        for nm in ["flash", "pattern", "redlight"]:
            idx = names.index(nm)
            if idx < len(vec) and vec[idx] == 1 and nm in PRED_TEXT_MAP:
                pred_texts.append(PRED_TEXT_MAP[nm])

        out_records.append({
            "frame": gp["frame"],
            "start": gp["start"],
            "end": gp["end"],
            "labels": label_kv,                 # {"phf":1,...,"pattern":1,"redlight":1}
            "수치_기반_결과": list(dict.fromkeys(cv_texts)),
            "예측_기반_결과": list(dict.fromkeys(pred_texts)),
        })
    return out_records