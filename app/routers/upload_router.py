import os
from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
from app.cruds.interpolation import conversion, get_frame_indices, extract_frame_ranges
from app.cruds.clip_position import clip_position_data

router = APIRouter(prefix="/nova/dashboard/video/upload") 

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_video(video: UploadFile = File(...)):
    try:
        # 1. video_id 및 경로 구성
        video_id = os.path.splitext(video.filename)[0]
        video_dir = os.path.join(UPLOAD_DIR, video_id)
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        # 2. 파일 저장
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 3. frames 폴더 확인
        frame_dir = os.path.join(video_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True) 
        if not os.path.exists(frame_dir):
            raise HTTPException(status_code=400, detail=f"frames 디렉토리가 없습니다: {frame_dir}")

        # 4. overlays 폴더 준비
        output_dir = os.path.join(video_dir, "overlays")
        os.makedirs(output_dir, exist_ok=True)

        for fname in os.listdir(output_dir):
            if fname.endswith(".mp4"):
                file_path = os.path.join(output_dir, fname)
                os.remove(file_path)
                print(f"[CLEAN] 기존 overlay 제거됨 → {file_path}")
                
        # 5. 프레임 목록 및 구간 추출 (소연이랑 연동 후 수정)
        frame_indices = get_frame_indices(frame_dir)
        frame_ranges = extract_frame_ranges(frame_indices)

        # 6. clip_position_Data
        clips = [
            [f"frame_{i:04d}.jpg" for i in range(start, end + 1)]
            for (start, end) in frame_ranges
        ]

        clip_data = clip_position_data(clips, video_path)
        fps = clip_data["fps"]

        # 7. 보간 처리
        generated_videos = []

        BASE_URL = "http://127.0.0.1:8000"

        for idx, clip in enumerate(clip_data["clips"]):
            frames = clip["frames"]
            start_idx = int(frames[0].split("_")[-1].split(".")[0])
            end_idx = int(frames[-1].split("_")[-1].split(".")[0])
            frame_range = list(range(start_idx, end_idx + 1))

            temp_root = os.path.join(video_dir, f"temp_{idx:04d}")
            result = conversion(
                video_id=video_id,
                original_video=video_path,
                frame_indices=frame_range,
                frame_root=frame_dir,
                output_root=output_dir,
                temp_root=temp_root,
                start_time=clip["start"]
            )

            if result is None:
                print(f"[SKIPPED] 프레임 {start_idx}-{end_idx}는 보간 실패로 스킵됨")
                continue

            start_frame = frame_range[0]
            end_frame = frame_range[-1]

            filename = os.path.basename(result['overlay_path'])
            video_url = f"{BASE_URL}/static/{video_id}/overlays/{filename}"
            generated_videos.append({
                "overlaySrc": video_url,
                "start": result["start"],
                "end": round(result["start"] + result["duration"], 3)
            })

        return {
            "message": f"'{video.filename}' 업로드 및 보간 완료",
            "video_id": video_id,
            "fps": clip_data["fps"],
            "overlays": generated_videos,
            "outputSrc": f"{BASE_URL}/static/{video_id}/{video_id}.mp4"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[업로드 및 보간 실패] {str(e)}")