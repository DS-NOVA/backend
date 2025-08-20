import os
import traceback
from pathlib import Path
from typing import List, Tuple, Dict
import sys, json
import asyncio
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.cruds.interp_crud import (
    group_consecutive_ones, assemble_interpolated_video, merge_ones_with_gap# ← ADD
)
# ✅ 검출 모델은 수정하지 않는 조건이므로, 그대로 import
from app.routers.pipeline_predict import predict_pipeline
from app.cruds.metrics_crud import run_cv_metrics, dump_json


router = APIRouter(prefix="/nova/dashboard/video/upload")

# =========[ 경로/상수 ]=========
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # C:\nova
STATIC_DIR   = PROJECT_ROOT / "static"                   # C:\nova\backend\static
UPLOAD_DIR   = STATIC_DIR / "uploads"                    # C:\nova\backend\static\uploads
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

FRAME_EXT = ".png"
BASE_URL  = "http://127.0.0.1:8000"  # 정적파일 마운트: /static


# =========[ 유틸 함수들 ]=========

def extract_frames(video_path: Path, frames_dir: Path, frame_ext: str = FRAME_EXT) -> tuple[int, float]:
    """
    원본 영상을 frames_dir에 frame_XXXX.png로 추출한다.
    return: (총 프레임 수, fps)
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 이전 잔여 프레임 정리
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

    # 일부 코덱/컨테이너에서 CAP_PROP_FRAME_COUNT가 0으로 나올 수 있어 idx로 보정
    if total_frames == 0:
        total_frames = idx

    return total_frames, float(fps)



def get_interpolation_ranges(frames_data: List[dict]) -> List[Tuple[int, int]]:
    """
    flash/pattern/redlight 중 하나라도 1인 프레임을 연속 구간으로 묶는다.
    예: [2,3,4,10,11] -> [(2,4),(10,11)]
    """
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

def format_detection_result(
    result_obj: dict,
    prepend_flags: Dict[str, List[int]] | None = None,
    prepend: bool = True
) -> List[dict]:
    flags = result_obj["flags"]
    num_frames = result_obj["num_frames"]
    base_names = ["flash", "pattern", "redlight"]
    extra_names = list(prepend_flags.keys()) if prepend_flags else []
    ordered_names = (extra_names + base_names) if prepend else (base_names + extra_names)

    def _at(arr, i, default=0):
        return int(arr[i]) if (arr and i < len(arr)) else default

    frames_data = []
    for i in range(num_frames):
        row = {"frame": i}
        for name in extra_names:
            row[name] = _at(prepend_flags[name], i, 0)
        for name in base_names:
            row[name] = _at(flags[name], i, 0)
        frames_data.append(row)
    return frames_data

# === (ADD) 프론트용 그래프 데이터로 변환: frame→range ===
def _make_graph_points_per_frame(frames_data: list[dict], fps: float, cv_label_order: list[str]) -> list[dict]:
    """
    frames_data: [{"frame": i, <labels...>}, ...]  # format_detection_result의 결과
    return: [{"start": sec, "end": sec, "labels": [9개]}, ...]  # 모든 프레임에 대해 1:1
    labels 순서 = cv_label_order(6) + ["flash","pattern","redlight"]
    """
    names = list(cv_label_order) + ["flash", "pattern", "redlight"]
    n = len(frames_data)
    if n == 0 or fps <= 0:
        return []

    out = []
    for i in range(n):
        # 각 라벨을 0/1로 강제
        vec = [int(frames_data[i].get(nm, 0)) for nm in names]
        start_sec = round(i / fps, 3)
        end_sec   = round((i + 1) / fps, 3)   # 프레임 i의 종료 시각을 i+1로
        out.append({"frame" : i, "start": start_sec, "end": end_sec, "labels": vec})
    return out


# =========[ 메인 엔드포인트 ]=========

@router.post("/")
async def upload_video(video: UploadFile = File(...)):
    try:
        # 1) 업로드/저장 경로 구성
        video_id   = Path(video.filename).stem
        video_dir  = UPLOAD_DIR / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / f"{video_id}.mp4"   # static/uploads/<id>/<id>.mp4
        raw_path   = RAW_DIR / f"{video_id}.mp4"     # data/raw/<id>.mp4

        # 2) 파일 저장 (uploads, raw)
        file_bytes = await video.read()
        (video_path).write_bytes(file_bytes)
        (raw_path).write_bytes(file_bytes)

        # 3) 출력 디렉토리 준비
        overlays_dir = video_dir / "overlays"
        frames_dir   = video_dir / "frames"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # 기존 overlay mp4 정리
        for p in overlays_dir.glob("*.mp4"):
            p.unlink(missing_ok=True)

        # 4) 프레임 추출 (frame_XXXX.png)
        total_frames, fps = extract_frames(video_path=video_path, frames_dir=frames_dir, frame_ext=FRAME_EXT)
        if total_frames == 0:
            raise HTTPException(status_code=400, detail=f"프레임 추출 실패: {video_path}")
        
        # 5) 검출 모델 실행 → flag 결과를 그래프용 JSON으로 변환
        debug_dir = UPLOAD_DIR / video_id / "cv_debug" 

        print(f"[DEBUG] predict & cv_metrics 호출: filename={video_id}.mp4")
        predict_task = asyncio.to_thread(predict_pipeline, f"{video_id}.mp4")               # RAW_DIR 기준
        cv_task = asyncio.to_thread(
            run_cv_metrics,
            RAW_DIR / f"{video_id}.mp4",
            fps=fps,
            debug_save_dir=debug_dir,           # ← 디버그 폴더 전달 (유지)
            debug_draw="both",
            save_bbox_outline=True
        )      # metrics_crud.run_cv_metrics(Path)
        predict_result, cv_flags = await asyncio.gather(predict_task, cv_task)

        if predict_result.get("status") not in ("saved", "skipped"):
            raise HTTPException(status_code=500, detail="예측 처리 실패")
        
        print(dict(cv_flags))  #itc 가이드라인 결과
        print("[DEBUG] predict 실행 결과(status):", predict_result) #검출 모델 결과
        print("[DEBUG] num_frames:", predict_result["result"]["num_frames"])
        print("[DEBUG] flash sum:", sum(predict_result["result"]["flags"]["flash"]))
        print("[DEBUG] pattern sum:", sum(predict_result["result"]["flags"]["pattern"]))
        print("[DEBUG] redlight sum:",sum(predict_result["result"]["flags"]["redlight"]))

        sys.stdout.flush()

        cv_json_path = UPLOAD_DIR / video_id / "cv_metrics.json"
        dump_json(cv_flags, cv_json_path, fps=fps)
        cv_label_order = list(cv_flags.keys())
        
        frames_data = format_detection_result(
            predict_result["result"],
            prepend_flags=cv_flags,
            prepend=True
        )  # graphData

        graph_points = _make_graph_points_per_frame(frames_data, fps=fps, cv_label_order=cv_label_order)

        risky_mask   = [1 if (fd["flash"] or fd["pattern"] or fd["redlight"]) else 0 for fd in frames_data]
        risky_ranges = group_consecutive_ones(risky_mask, min_len=2)
        bridged_ranges = merge_ones_with_gap(risky_mask, max_gap=3, min_len=2)

        insert_mask = [0] * max(0, len(frames_data) - 1)
        for (s, e) in risky_ranges:
            for k in range(s, e):
                insert_mask[k] = 1


        print("[DEBUG] risky_ranges:", risky_ranges)
        print("[DEBUG] bridged_ranges (gap<=3 merged):", bridged_ranges)    
        print("[DEBUG] graph_ranges: ", graph_points)

        if not bridged_ranges:
            return {
                "message": f"'{video.filename}' 업로드 완료 (위험 프레임 없음, 보간 미수행)",
                "video_id": video_id,
                "fps": fps,
                "overlays": [],
                "graphData": graph_points,
                "cvLabelOrder": cv_label_order,
                "cvMetricsJson": f"{BASE_URL}/static/uploads/{video_id}/cv_metrics.json",
                "outputSrc": f"{BASE_URL}/static/uploads/{video_id}/{video_id}.mp4",
                "interpolatedSpans": [],
                "riskyRanges": []
            }
        
        # 7) 보간 실행: t~t+1 구간에서 0.5 프레임 삽입 → mp4 조각 생성
        full = assemble_interpolated_video(
            video_id=video_id,
            fps=fps,
            base_dir=str(PROJECT_ROOT),
            frame_ext=FRAME_EXT,
            checkpoint_abs_path="C:/nova/IFRNet-main/IFRNet_Vimeo90K.pth",
            insert_mode="mask",            # mid 삽입은 mask로 제어
            risky_ranges=bridged_ranges,   # 밝기/적응 임계치/표시는 '브릿지 구간' 기준
            risky_mask=insert_mask,
            # ↓↓↓ 고정값 대신 적응형 사용
            use_adaptive_thresh=True,          # ← 이거 켜기
            # (원하면 미세조정)
            adaptive_margin=-10,               # 평균보다 조금 더 낮게 잡아 넓게 마스킹
            adaptive_clamp_low=50,
            adaptive_clamp_high=250,
            mask_atten_factor=0.6,             # 얼마나 어둡게(작을수록 더 어둡게)
            mask_blur_ksize=31,
            save_intermediate=True,
            dump_dir_name="risky_dump",
            overwrite_dump=True,      # 마스크 블러 (홀수)
        )
        full_url = f"{BASE_URL}/static/uploads/{video_id}/overlays/{Path(full['output_path']).name}"

        interpolated_spans = []


        for (s, e) in risky_ranges:
            # 구간 내 인접쌍 (s,s+1), ... (e-1,e) 에 대해 중간시간대 강조
            # 여기선 mid를 “k와 k+1 사이 시간”으로 표기(시각화 전용)
            for k in range(s, e):
                start_t = round(k / fps, 3)
                end_t   = round((k + 1) / fps, 3)
                interpolated_spans.append({"start": start_t, "end": end_t})
                
        risky_ranges_sec = [
            {"start": round(s / fps, 3), "end": round((e + 1) / fps, 3)}
            for (s, e) in bridged_ranges
        ]

        return {
            "message": f"'{video.filename}' 업로드 및 보간(단일 mp4) 완료",
            "video_id": video_id,
            "fps": fps,
            "graphData": graph_points,
            "cvLabelOrder": cv_label_order,
            "cvMetricsJson": f"{BASE_URL}/static/uploads/{video_id}/cv_metrics.json",
            "interpolatedSpans": interpolated_spans,   # ← 보간 페어(초)
            "riskyRanges": risky_ranges_sec,           # ← 브릿지 구간(초) ★ 프론트 표시에 사용
            "riskyRangesIdx": bridged_ranges,          # ← (옵션) 디버깅/로그용 프레임 인덱스 원본
            "outputSrc": f"{BASE_URL}/static/uploads/{video_id}/{video_id}.mp4",
            "outputSrcInterpolated": full_url
        }


    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"[업로드 및 보간 실패] {str(e)}")
