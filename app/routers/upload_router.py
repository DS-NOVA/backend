import os
import traceback
from pathlib import Path
from typing import List, Tuple, Dict
import sys, json
import asyncio
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from app.security.auth import get_current_user
from app.db.database import get_db
from sqlalchemy.orm import Session
from app.models.video import Video
from datetime import datetime
import time
from app.cruds.interp_crud import (
assemble_interpolated_video, merge_ones_with_gap, RampVSParams
)
#검출 모델은 수정하지 않는 조건이므로, 그대로 import
from app.routers.pipeline_predict import predict_pipeline
from app.cruds.metrics_crud import run_cv_metrics, dump_json
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    try:
        # 아시아 CID 폰트(시스템에 있을 수도, 없을 수도). 실패해도 괜찮게 처리.
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        _CID_AVAILABLE = True
    except Exception:
        _CID_AVAILABLE = False
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

router = APIRouter(prefix="/nova/dashboard/video/upload")

# 경로
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # C:\nova
STATIC_DIR   = PROJECT_ROOT / "static"
UPLOAD_DIR   = STATIC_DIR / "uploads"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

FRAME_EXT = ".png"
BASE_URL  = "http://127.0.0.1:8000"  # 정적파일 마운트: /static


# 유틸 함수 
# 비디오 파일에서 프레임을 추출해 frame_dir에 저장
# 총 프레임 수, fps 반환 
def extract_frames(video_path: Path, frames_dir: Path, frame_ext: str = FRAME_EXT) -> tuple[int, float]:
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

# =====[ PDF 한글 폰트 등록 헬퍼 ]=====
def _register_korean_font() -> str:
    """
    한글 지원 폰트를 등록하고 그 폰트명을 반환.
    우선순위:
      1) STATIC_DIR/fonts 안의 TTF/OTF (NanumGothic, NotoSansKR, Pretendard 등)
      2) ReportLab CID 폰트(HYGoThic-Medium) - 환경에 따라 미지원일 수 있음
      3) 실패 시 Helvetica (한글은 깨질 수 있음)
    """
    if not REPORTLAB_AVAILABLE:
        return "Helvetica"

    candidates = [
        STATIC_DIR / "fonts" / "PretendardGOV-ExtraBold.ttf",
        STATIC_DIR / "fonts" / "PretendardGOV-Bold.ttf",
        STATIC_DIR / "fonts" / "PretendardGOV-Regular.ttf",
        STATIC_DIR / "fonts" / "PretendardGOV-Medium.ttf",
        STATIC_DIR / "fonts" / "PretendardGOV-Light.ttf"
    ]
    for path in candidates:
        try:
            if path.exists():
                pdfmetrics.registerFont(TTFont("KoreanUI", str(path)))
                return "KoreanUI"
        except Exception:
            pass

    # CID 폰트(환경에 따라 동작). 실패해도 전파하지 않고 넘어간다.
    try:
        if _CID_AVAILABLE:
            pdfmetrics.registerFont(UnicodeCIDFont("HYGoThic-Medium"))
            return "HYGoThic-Medium"
    except Exception:
        pass

    # 최후 fallback
    return "Helvetica"

# 1. upload router
@router.post("/")
async def upload_video(
    video: UploadFile = File(...),
    brightness: float = Form(0.75),
    saturation: float = Form(0.75),
    gamma_max: float = Form(1.0),
    hold_ratio: float = Form(0.05),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)):
    start_time = time.time()
    try:

        # 사용자가 입력한 밝기, 채도 값 
        requested_params = {
            "brightness": float(brightness),
            "saturation": float(saturation),
            "hold_ratio": float(hold_ratio),
            "gamma_max": float(gamma_max),
        }
        v_min = float(max(0.0, min(1.0, brightness)))
        s_min = float(max(0.0, min(1.0, saturation)))
        hr    = float(max(0.0, min(0.8, hold_ratio)))
        gmax  = float(max(1.0, gamma_max))
        applied_params = {"v_min": v_min, "s_min": s_min, "hold_ratio": hr, "gamma_max": gmax}

        
        # 경로 세팅 
        file_key   = Path(video.filename).stem
        video_id   = Path(video.filename).stem
        video_dir  = UPLOAD_DIR / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / f"{video_id}.mp4"
        raw_path   = RAW_DIR / f"{video_id}.mp4"

        # 파일 저장 
        file_bytes = await video.read()
        (video_path).write_bytes(file_bytes)
        (raw_path).write_bytes(file_bytes)

        # 출력 디렉토리
        overlays_dir = video_dir / "overlays"
        frames_dir   = video_dir / "frames"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for p in overlays_dir.glob("*.mp4"):
            p.unlink(missing_ok=True)

        # 프레임 추출하기 
        total_frames, fps = extract_frames(video_path=video_path, 
                                           frames_dir=frames_dir, frame_ext=FRAME_EXT)
        if total_frames == 0:
            raise HTTPException(status_code=400, detail=f"프레임 추출 실패: {video_path}")
        
        # 검출 모델 및 openCV (itc 가이드라인) 결과 동시 실행 
        predict_task = asyncio.to_thread(predict_pipeline, f"{video_id}.mp4")
        cv_task = asyncio.to_thread(
            run_cv_metrics,
            RAW_DIR / f"{video_id}.mp4",
            fps=fps,
            debug_save_dir=None,
            debug_draw="none",
            save_bbox_outline=False
        )
        predict_result, cv_flags = await asyncio.gather(predict_task, cv_task)
        if predict_result.get("status") not in ("saved", "skipped"):
            raise HTTPException(status_code=500, detail="예측 처리 실패")

        print("[DEBUG] predict 실행 결과(status):", predict_result)
        sys.stdout.flush()

        # openCV 결과 저장하기 -> 프론트로 전송
        cv_json_path = UPLOAD_DIR / video_id / "cv_metrics.json"
        dump_json(cv_flags, cv_json_path, fps=fps)
        cv_label_order = list(cv_flags.keys())
        cv_union_by_label = cv_union_vector_by_label(cv_flags, cv_label_order)
    

        frames_data = format_detection_result(
            predict_result["result"],
            prepend_flags=cv_flags,
            prepend=True
        )

        # 그래프 시각화 배열 생성 
        graph_points = _make_graph_points_per_frame(frames_data, fps=fps, cv_label_order=cv_label_order)

        guideline_summary = {
            "flash":   {"violated": sum(predict_result["result"]["flags"]["flash"])   > 0},
            "pattern": {"violated": sum(predict_result["result"]["flags"]["pattern"]) > 0},
            "redlight":{"violated": sum(predict_result["result"]["flags"]["redlight"])> 0}
        }

        frame_labels_pdf_path = save_local_frame_labels(
            video_dir, graph_points, cv_label_order, make_pdf=True
        )


        # 하나의 프레임에 flash, pattern, redlight이 하나 이상인 경우 1 (OR 연산)
        risky_mask = [1 if (fd["flash"] or fd["pattern"] or fd["redlight"]) else 0 for fd in frames_data]
        n_frames = len(frames_data)

        # max_gap 사이의 0 프레임을 허용 (risky_mask 등장 후 5번째 이내에 1이 등장할 경우 포함)
        bridged_ranges = merge_ones_with_gap(risky_mask, max_gap=5, min_len=1) 

        #위험이 전혀 없으면(모두 0) 보간, 밝기, 채도 변화 없이 반환하기 
        if sum(risky_mask) == 0:  
            result_json = {
                "video_id": file_key,
                "fps": fps,
                "graphData": graph_points, # 프레임 단위 시각화 데이터
                "cvLabelOrder": cv_label_order, # cv 라벨 이름 순서
                "cvUnionByLabel": cv_union_by_label, # cv OR 연산 (가이드라인 6가지가 한 번이라도 1이 나온 적이 있는지)
                "requestedParams": requested_params, # 사용자가 선택한 밝기, 채도
                "appliedParams": applied_params, # 실제 파라미터 
                "cvMetricsJson": f"{BASE_URL}/static/uploads/{file_key}/cv_metrics.json", #openCV 결과 요약
                "interpolatedSpans": [], # 보간이 삽입된 구간
                "riskyRanges": [], # 위험 구간
                "inputSrc": f"{BASE_URL}/static/uploads/{file_key}/{file_key}.mp4", # 원본 영상
                "outputSrc": f"{BASE_URL}/static/uploads/{file_key}/{file_key}.mp4", # 변환된 영상 
                "outputSrcInterpolated": None, # 보간 결과가 없으므로 None
                "guidelineSummary": guideline_summary, # 검출 모델 결과 
                "frameLabelsJson": f"{BASE_URL}/static/uploads/{file_key}/frame_labels.json", # 프레임 별 라벨 요약
                "frameLabelsPdf":  (f"{BASE_URL}/static/uploads/{file_key}/frame_labels.pdf" # 프레임 별 라벨 요약 pdf
                                    if frame_labels_pdf_path else None)
            }
            (video_dir / "result.json").write_text(json.dumps(result_json, ensure_ascii=False), encoding="utf-8")
            print(f"위험구간 없는 경우: {time.time() - start_time:.2f}초")
            
            return {
                "message": f"'{video.filename}' 업로드 완료 (위험 프레임 없음, 보간/램프 미수행)",
                "video_id": file_key,
                "fps": fps,
                "graphData": graph_points,
                "cvLabelOrder": cv_label_order,
                "cvUnionByLabel": cv_union_by_label,
                "requestedParams": requested_params,
                "appliedParams": applied_params,
                "cvMetricsJson": f"{BASE_URL}/static/uploads/{video_id}/cv_metrics.json",
                "interpolatedSpans": [],
                "riskyRanges": [],
                "inputSrc": f"{BASE_URL}/static/uploads/{file_key}/{file_key}.mp4",
                "outputSrc": f"{BASE_URL}/static/uploads/{video_id}/{video_id}.mp4",
                "guidelineSummary": guideline_summary,
                "frameLabelsJson": f"{BASE_URL}/static/uploads/{file_key}/frame_labels.json", 
                "frameLabelsPdf":  (f"{BASE_URL}/static/uploads/{file_key}/frame_labels.pdf"  
                                    if frame_labels_pdf_path else None)
            }

        # 위험 구간이 있는 경우
        # 인접 프레임이 모두 1이며, 위험 구간 (bridged_ranges) 에 포함 되어 있을 때만 보간 
        insert_mask = [0] * max(0, n_frames - 1)
        insert_pairs_idx = []

        def _in_any_range(i: int) -> bool:
            for s, e in bridged_ranges:
                if s <= i and (i + 1) <= e:
                    return True
            return False

        for i in range(n_frames - 1):
            if risky_mask[i] == 1 and risky_mask[i + 1] == 1 and _in_any_range(i):
                insert_mask[i] = 1
                insert_pairs_idx.append((i, i + 1))

        #보간 실행: 위험구간 내 (i,i+1)에만 0.5 프레임 삽입, bridged_ranges에 밝기, 채도 조절 
        full = assemble_interpolated_video(
            video_id=video_id,
            fps=fps,
            base_dir=str(PROJECT_ROOT),
            frame_ext=FRAME_EXT,
            checkpoint_abs_path="C:/nova/IFRNet-main/IFRNet_Vimeo90K.pth",
            insert_mode="mask",
            risky_ranges=bridged_ranges,    # 밝기/채도 램프 적용 범위(0 포함, 위험구간 내부만)
            risky_mask=insert_mask,         # 보간 삽입 위치(위험구간 내부 쌍만)
            use_vs_ramp=True,
            ramp_params=RampVSParams(
                mode="both",
                v_min=v_min, # 밝기
                s_min=s_min, # 채도
                gamma_max=gmax, # 감마
                hold_ratio=hr # 중앙부 유지 비율
            ),
            save_intermediate=True, # 디버깅시 사용 
            dump_dir_name="risky_dump",
            overwrite_dump=True,
            stream_from_raw=True,
            raw_video_path=RAW_DIR / f"{video_id}.mp4"
        )
        full_url = f"{BASE_URL}/static/uploads/{video_id}/overlays/{Path(full['output_path']).name}"

        # 프론트 툴팁 
        interpolated_spans = [{"start": round(s / fps, 3), "end": round(e / fps, 3)} for (s, e) in insert_pairs_idx]

        # 프론트 표시용 위험 구간(초)
        risky_ranges_sec = []
        for (s, e) in bridged_ranges:
            label_names = list(cv_label_order) + ["flash", "pattern", "redlight"]
            labels_in_span = _active_labels_in_span(frames_data, label_names, s, e)  # 예: ['php','psp']
            risky_ranges_sec.append({
                "start": round(s / fps, 3),
                "end":   round((e + 1) / fps, 3),   # 프레임 e의 끝까지 포함
                "labels": labels_in_span            # 텍스트 대신 라벨 코드만
            })

        result_json = {
            "video_id": file_key,
            "fps": fps,
            "graphData": graph_points,
            "cvLabelOrder": cv_label_order,
            "cvUnionByLabel": cv_union_by_label,
            "requestedParams": requested_params,
            "appliedParams": applied_params,
            "cvMetricsJson": f"{BASE_URL}/static/uploads/{file_key}/cv_metrics.json",
            "interpolatedSpans": interpolated_spans,
            "riskyRanges": risky_ranges_sec,
            "riskyRangesIdx": bridged_ranges,
            "inputSrc": f"{BASE_URL}/static/uploads/{file_key}/{file_key}.mp4",
            "outputSrc": f"{BASE_URL}/static/uploads/{file_key}/{file_key}.mp4",
            "outputSrcInterpolated": full_url,
            "guidelineSummary": guideline_summary,
            "frameLabelsJson": f"{BASE_URL}/static/uploads/{file_key}/frame_labels.json",
            "frameLabelsPdf":  (f"{BASE_URL}/static/uploads/{file_key}/frame_labels.pdf"  
                                    if frame_labels_pdf_path else None)
        }
        (video_dir / "result.json").write_text(json.dumps(result_json, ensure_ascii=False), encoding="utf-8")
        print(f"전체 걸리는 시간: {time.time() - start_time:.2f}초")

        return {
            "message": f"'{video.filename}' 업로드 및 보간(단일 mp4) 완료",
            "video_id": file_key,
            "fps": fps,
            "graphData": graph_points,
            "cvLabelOrder": cv_label_order,
            "cvUnionByLabel": cv_union_by_label,
            "requestedParams": requested_params,
            "appliedParams": applied_params,
            "cvMetricsJson": f"{BASE_URL}/static/uploads/{video_id}/cv_metrics.json",
            "interpolatedSpans": interpolated_spans,
            "riskyRanges": risky_ranges_sec,
            "riskyRangesIdx": bridged_ranges,
            "inputSrc": f"{BASE_URL}/static/uploads/{file_key}/{file_key}.mp4",
            "outputSrc": f"{BASE_URL}/static/uploads/{video_id}/{video_id}.mp4",
            "outputSrcInterpolated": full_url,
            "guidelineSummary": guideline_summary,
            "frameLabelsJson": f"{BASE_URL}/static/uploads/{file_key}/frame_labels.json",
            "frameLabelsPdf":  (f"{BASE_URL}/static/uploads/{file_key}/frame_labels.pdf"  
                                    if frame_labels_pdf_path else None)
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"[업로드 및 보간 실패] {str(e)}")


# === helpers: CV metrics 합치기 ===
# 라벨중 하나라도 1이 있으면 1 -> 그래프 가이드라인에 활용
def cv_union_vector_by_label(cv_flags: dict[str, list[int]], label_order: list[str]) -> list[int]:
    if not cv_flags or not label_order:
        return []
    out = []
    for name in label_order:
        arr = cv_flags.get(name) or []
        out.append(1 if any(int(x) != 0 for x in arr) else 0)
    return out

# frame_labels.json, pdf 요약 생성성    
def save_local_frame_labels(
    video_dir: Path,
    graph_points: list[dict],
    cv_label_order: list[str],
    make_pdf: bool = True
) -> tuple[Path, Path | None]:
    """
    graph_points: _make_graph_points_per_frame() 결과(프레임별 start/end + n차원 labels)
    - 모든 라벨이 0인 프레임은 제외해서 저장
    - frame_labels.json과 (옵션) frame_labels.pdf 생성
    return: (json_path, pdf_path_or_None)
    """
    names = list(cv_label_order) + ["flash", "pattern", "redlight"]

    CV_TEXT_MAP = {
        "phf": "잠재적으로 해로운 섬광이 있습니다.",
        "phr": "잠재적으로 해로운 적색 섬광이 있습니다.",
        "pfs": "잠재적으로 해로운 섬광이 깜빡거립니다.",
        "ps" : "잠재적으로 해로운 깜빡임이 지속됩니다.",
        "php": "잠재적으로 해로운 패턴이 있습니다.",
        "psp": "잠재적으로 해로운 패턴이 일정 면적을 차지합니다.",
    }
    PRED_TEXT_MAP = {
        "flash":    "해로운 섬광이 검출되었습니다.",
        "pattern":  "해로운 패턴이 검출되었습니다.",
        "redlight": "해로운 적색 섬광이 검출되었습니다.",
    }

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

    # PDF 저장(옵션)
    pdf_path = None
    if make_pdf:
        pdf_path = video_dir / "frame_labels.pdf"
        try:
            export_frame_labels_pdf(out_records, pdf_path, CV_TEXT_MAP, PRED_TEXT_MAP)
        except Exception:
            # reportlab 미설치/오류 시 PDF는 생략
            pdf_path = None

    return json_path, pdf_path


def export_frame_labels_pdf(records: list[dict], pdf_path: Path,
                            cv_text_map: dict[str, str],
                            pred_text_map: dict[str, str]):
    """ReportLab로 표 형태 PDF 생성 (A4 가로, 좁은 여백, 한글 폰트 임베드, 자동 줄바꿈)."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab not available")

    # 페이지/여백/폰트
    use_font = _register_korean_font()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=14, rightMargin=14, topMargin=14, bottomMargin=14,
        title="프레임 라벨 요약"
    )
    styles = getSampleStyleSheet()
    # 기본 스타일에 폰트 적용
    for k in ["Title", "Heading1", "Heading2", "Heading3", "BodyText"]:
        if k in styles.byName:
            styles[k].fontName = use_font

    # 표 셀에서 더 작은 글씨/줄바꿈을 위한 스타일
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["BodyText"],
        fontName=use_font,
        fontSize=8,
        leading=10,
    )
    small_head = ParagraphStyle(
        "SmallHead",
        parent=styles["Heading2"],
        fontName=use_font,
        fontSize=11,
        leading=13,
    )

    story = []
    story.append(Paragraph("프레임 라벨 요약", styles["Title"]))
    story.append(Spacer(1, 6))

    # === Legend (CV / Pred) ===
    story.append(Paragraph("코드 설명 (CV 규칙)", small_head))
    cv_rows = [[Paragraph("코드", cell_style), Paragraph("설명", cell_style)]]  # ← 변경
    for k, v in sorted(cv_text_map.items()):
        cv_rows.append([Paragraph(k, cell_style), Paragraph(v, cell_style)])

    tbl_cv = Table(cv_rows, colWidths=[doc.width * 0.12, doc.width * 0.88])
    tbl_cv.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("FONTNAME", (0,0), (-1,-1), use_font),  # ← 추가: 혹시 문자열이 남아도 폰트 강제
    ]))
    story.append(tbl_cv)
    story.append(Spacer(1, 6))

    story.append(Paragraph("코드 설명 (검출 결과)", small_head))
    pred_rows = [[Paragraph("코드", cell_style), Paragraph("설명", cell_style)]]  # ← 변경
    for k, v in sorted(pred_text_map.items()):
        pred_rows.append([Paragraph(k, cell_style), Paragraph(v, cell_style)])

    tbl_pred = Table(pred_rows, colWidths=[doc.width * 0.12, doc.width * 0.88])
    tbl_pred.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("FONTNAME", (0,0), (-1,-1), use_font),  # ← 추가
    ]))

    story.append(tbl_pred)
    story.append(Spacer(1, 10))

    # === Records ===
    if not records:
        story.append(Paragraph("활성 라벨이 없습니다(모든 프레임이 0).", styles["BodyText"]))
        doc.build(story)
        return

    header = ["프레임", "구간(s)", "활성 라벨", "수치 기반 결과", "예측 기반 결과"]
    table_rows = [[Paragraph(h, cell_style) for h in header]]

    for r in records:
        active_labels = [k for k, v in r["labels"].items() if v == 1]
        table_rows.append([
            Paragraph(str(r["frame"]), cell_style),
            Paragraph(f"{r['start']}–{r['end']}", cell_style),
            Paragraph(", ".join(active_labels) if active_labels else "-", cell_style),
            Paragraph(" / ".join(r["수치_기반_결과"]) if r["수치_기반_결과"] else "-", cell_style),
            Paragraph(" / ".join(r["예측_기반_결과"]) if r["예측_기반_결과"] else "-", cell_style),
        ])

    # 넓은 표: 가로폭을 꽉 쓰도록 가변 너비 설정 (A4 가로)
    avail = doc.width
    col_widths = [avail * 0.08, avail * 0.12, avail * 0.20, avail * 0.30, avail * 0.30]

    tbl = Table(table_rows, repeatRows=1, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("FONTSIZE", (0,0), (-1,-1), 8),
    ]))

    story.append(tbl)
    doc.build(story)
