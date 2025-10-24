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
import time
from app.cruds.interp_crud import (
assemble_interpolated_video, merge_ones_with_gap, RampVSParams
)
from app.routers.pipeline_predict import predict_pipeline
from app.cruds.metrics_crud import run_cv_metrics, dump_json
from app.utils.pdf_report import export_frame_labels_pdf
from app.utils.frame_labels import CV_TEXT_MAP, PRED_TEXT_MAP
from app.utils.json_report import export_frame_labels_json
from app.utils.frame_format import extract_frames, get_interpolation_ranges, format_detection_result, _make_graph_points_per_frame, _active_labels_in_span, cv_union_vector_by_label

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

        frame_labels_json_path, frame_labels_pdf_path = save_local_frame_labels(
            video_dir, graph_points, cv_label_order, make_pdf=True
        )


        # 하나의 프레임에 flash, pattern, redlight이 하나 이상인 경우 1 (OR 연산)
        risky_mask = [1 if (fd["flash"] or fd["pattern"] or fd["redlight"]) else 0 for fd in frames_data]
        n_frames = len(frames_data)

        # max_gap 사이의 0 프레임을 허용 (risky_mask 등장 후 5번째 이내에 1이 등장할 경우 포함)
        bridged_ranges = merge_ones_with_gap(risky_mask, max_gap=5, min_len=1) 

        #위험이 전혀 없으면(모두 0) 보간, 밝기, 채도 변화 없이 반환하기 
        if sum(risky_mask) == 0:  
            result_json = build_result_payload(
                file_key=file_key,
                video_id=video_id,
                fps=fps,
                graph_points=graph_points,
                cv_label_order=cv_label_order,
                cv_union_by_label=cv_union_by_label,
                requested_params=requested_params,
                applied_params=applied_params,
                cv_json_path=cv_json_path,
                guideline_summary=guideline_summary,
                frame_labels_pdf_path=frame_labels_pdf_path,
                risky_mask=risky_mask,
                bridged_ranges=bridged_ranges,
                frames_data=frames_data,
                base_url=BASE_URL
            )
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
        full = await assemble_interpolated_video(
            video_id=video_id,
            fps=fps,
            base_dir=str(PROJECT_ROOT),
            frame_ext=FRAME_EXT,
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

        result_json = build_result_payload(
            file_key=file_key,
            video_id=video_id,
            fps=fps,
            graph_points=graph_points,
            cv_label_order=cv_label_order,
            cv_union_by_label=cv_union_by_label,
            requested_params=requested_params,
            applied_params=applied_params,
            cv_json_path=cv_json_path,
            guideline_summary=guideline_summary,
            frame_labels_pdf_path=frame_labels_pdf_path,
            risky_mask=risky_mask,
            bridged_ranges=bridged_ranges,
            frames_data=frames_data,
            base_url=BASE_URL
            )
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


# frame_labels.json, pdf 요약 생성 
def save_local_frame_labels(video_dir, graph_points, cv_label_order, make_pdf):
    json_path, out_records = export_frame_labels_json(video_dir, graph_points, cv_label_order)
    pdf_path = None

    if make_pdf:
        pdf_path = video_dir / "frame_labels.pdf"
        try:
            export_frame_labels_pdf(out_records, pdf_path, CV_TEXT_MAP, PRED_TEXT_MAP, STATIC_DIR)
        except Exception:
            # reportlab 미설치/오류 시 PDF는 생략
            pdf_path = None
    return json_path, pdf_path



def build_result_payload(
    file_key: str,
    video_id: str,
    fps: float,
    graph_points: list[dict],
    cv_label_order: list[str],
    cv_union_by_label: list[int],
    requested_params: dict,
    applied_params: dict,
    cv_json_path: Path,
    guideline_summary: dict,
    frame_labels_pdf_path: Path | None,
    risky_mask: list[int],
    bridged_ranges: list[tuple[int, int]],
    frames_data: list[dict],
    base_url: str = BASE_URL
) -> dict:
    """
    업로드 및 분석 결과를 정리해 프론트로 보낼 JSON payload 생성.
    위험 프레임 여부에 따라 분기 처리.
    """

    # 위험 구간이 전혀 없는 경우
    if sum(risky_mask) == 0:
        return {
            "video_id": file_key,
            "fps": fps,
            "graphData": graph_points,
            "cvLabelOrder": cv_label_order,
            "cvUnionByLabel": cv_union_by_label,
            "requestedParams": requested_params,
            "appliedParams": applied_params,
            "cvMetricsJson": f"{base_url}/static/uploads/{file_key}/cv_metrics.json",
            "interpolatedSpans": [],
            "riskyRanges": [],
            "inputSrc": f"{base_url}/static/uploads/{file_key}/{file_key}.mp4",
            "outputSrc": f"{base_url}/static/uploads/{file_key}/{file_key}.mp4",
            "outputSrcInterpolated": None,
            "guidelineSummary": guideline_summary,
            "frameLabelsJson": f"{base_url}/static/uploads/{file_key}/frame_labels.json",
            "frameLabelsPdf": (
                f"{base_url}/static/uploads/{file_key}/frame_labels.pdf"
                if frame_labels_pdf_path else None
            )
        }

    # 위험 구간이 존재하는 경우
    # 보간 위치와 위험 라벨 구간 구성
    n_frames = len(frames_data)
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

    interpolated_spans = [{"start": round(s / fps, 3), "end": round(e / fps, 3)} for (s, e) in insert_pairs_idx]

    risky_ranges_sec = []
    label_names = list(cv_label_order) + ["flash", "pattern", "redlight"]
    for (s, e) in bridged_ranges:
        labels_in_span = _active_labels_in_span(frames_data, label_names, s, e)
        risky_ranges_sec.append({
            "start": round(s / fps, 3),
            "end": round((e + 1) / fps, 3),
            "labels": labels_in_span
        })

    # 결과 JSON 생성
    return {
        "video_id": file_key,
        "fps": fps,
        "graphData": graph_points,
        "cvLabelOrder": cv_label_order,
        "cvUnionByLabel": cv_union_by_label,
        "requestedParams": requested_params,
        "appliedParams": applied_params,
        "cvMetricsJson": f"{base_url}/static/uploads/{file_key}/cv_metrics.json",
        "interpolatedSpans": interpolated_spans,
        "riskyRanges": risky_ranges_sec,
        "riskyRangesIdx": bridged_ranges,
        "inputSrc": f"{base_url}/static/uploads/{file_key}/{file_key}.mp4",
        "outputSrc": f"{base_url}/static/uploads/{file_key}/{file_key}.mp4",
        "outputSrcInterpolated": None,  # 보간된 영상 URL은 나중에 채움
        "guidelineSummary": guideline_summary,
        "frameLabelsJson": f"{base_url}/static/uploads/{file_key}/frame_labels.json",
        "frameLabelsPdf": (
            f"{base_url}/static/uploads/{file_key}/frame_labels.pdf"
            if frame_labels_pdf_path else None
        )
    }
