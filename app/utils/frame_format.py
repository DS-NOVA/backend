from pathlib import Path
from typing import List, Tuple, Dict
import cv2
from fastapi import HTTPException


def extract_frames(video_path: Path, frames_dir: Path, frame_ext: str) -> tuple[int, float]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob(f"frame_*{frame_ext}"):
        p.unlink(missing_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"영상을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = frames_dir / f"frame_{idx:04d}{frame_ext}"
        cv2.imwrite(str(out_path), frame)
        idx += 1
    cap.release()
    if total_frames == 0:
        total_frames = idx
    return total_frames, float(fps)

# 연속된 프레임 (flash/pattern/redlight) 중 하나라도 1이 있다면,
# [시작, 끝] 구간으로 묶어 리스트로 반환 
def get_interpolation_ranges(frames_data: List[dict]) -> List[Tuple[int, int]]:
    risky = [f["frame"] for f in frames_data if f["flash"] or f["pattern"] or f["redlight"]]
    if not risky:
        return []
    ranges = []
    start = prev = risky[0]
    for fr in risky[1:]:
        if fr == prev + 1:
            prev = fr
        else:
            ranges.append((start, prev))
            start = fr
            prev = fr
    ranges.append((start, prev))
    return ranges

# 프레임 인덱스, 각 라벨의 결과 (0/1), 결과 확률을 정규화하여 반환 
def format_detection_result(
    result_obj: dict,
    prepend_flags: Dict[str, List[int]] | None = None,
    prepend: bool = True
) -> List[dict]:
    flags = result_obj["flags"]
    probs = result_obj.get("probs", {})
    num_frames = result_obj["num_frames"]
    base_names = ["flash", "pattern", "redlight"]
    extra_names = list(prepend_flags.keys()) if prepend_flags else []
    def _at(arr, i, default=0):
        return int(arr[i]) if (arr and i < len(arr)) else default
    def _atf(arr, i, default=0.0): return float(arr[i]) if (arr and i < len(arr)) else default
    frames_data = []
    for i in range(num_frames):
        row = {"frame": i}
        for name in extra_names:
            row[name] = _at(prepend_flags[name], i, 0)
        for name in base_names:
            row[name] = _at(flags[name], i, 0)
        # 확률
        row["flash_prob"]    = _atf(probs.get("flash"), i, 0.0)
        row["pattern_prob"]  = _atf(probs.get("pattern"), i, 0.0)
        row["redlight_prob"] = _atf(probs.get("redlight"), i, 0.0)
        frames_data.append(row)
    return frames_data

# 프레임별 시작/끝, 모든 라벨의 0/1, 확률을 묶어
# 그래프 시각화용 배열 생성 
def _make_graph_points_per_frame(frames_data: list[dict], fps: float, cv_label_order: list[str]) -> list[dict]:
    names = list(cv_label_order) + ["flash", "pattern", "redlight"]
    n = len(frames_data)
    if n == 0 or fps <= 0:
        return []
    out = []
    for i in range(n):
        vec = [int(frames_data[i].get(nm, 0)) for nm in names]
        start_sec = round(i / fps, 3)
        end_sec   = round((i + 1) / fps, 3)
        out.append({"frame" : i, 
                    "start": start_sec, 
                    "end": end_sec, 
                    "labels": vec, 
                    "probs": {
                        "flash":   float(frames_data[i].get("flash_prob", 0.0)),
                        "pattern": float(frames_data[i].get("pattern_prob", 0.0)),
                        "redlight":float(frames_data[i].get("redlight_prob", 0.0)),
                    }
        })
    return out

# [시작, 끝] 구간에서 한 번이라도 1이 나온 라벨을 정렬하여 반환  (예: php, psp 등)
def _active_labels_in_span(
    frames_data: List[dict],
    label_names: List[str],   # 보낼 라벨들 (예: cv_label_order)
    s: int,
    e: int
) -> List[str]:
    if not frames_data or not label_names:
        return []
    s = max(0, s)
    e = min(e, len(frames_data) - 1)
    on = set()
    for i in range(s, e + 1):
        row = frames_data[i]
        for nm in label_names:
            if int(row.get(nm, 0)) != 0:
                on.add(nm)
    return sorted(on)

# 라벨 중 하나라도 1이 있으면 1 -> 그래프 가이드라인에 활용
def cv_union_vector_by_label(cv_flags: dict[str, list[int]], label_order: list[str]) -> list[int]:
    if not cv_flags or not label_order:
        return []
    out = []
    for name in label_order:
        arr = cv_flags.get(name) or []
        out.append(1 if any(int(x) != 0 for x in arr) else 0)
    return out