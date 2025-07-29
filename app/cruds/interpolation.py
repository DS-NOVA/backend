import subprocess
import os
import json
import re
import sys
import shutil
from dotenv import load_dotenv

load_dotenv()

#프레임을 숫자로 변경
def get_frame_indices(frame_dir):
    # frame_XXXX.jpg → XXXX 추출해서 숫자로 변환
    frame_files = os.listdir(frame_dir)
    indices = []
    for fname in frame_files:
        match = re.match(r"frame_(\d+)\.jpg", fname)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)

#프레임 묶기 -> 추후 수정 (-> 소연)
def extract_frame_ranges(frames):
    frames = sorted(frames)  # 혹시나 정렬 안된 경우 대비
    ranges = []

    start = frames[0]
    prev = frames[0]

    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            ranges.append((start, prev))
            start = f
            prev = f
    ranges.append((start, prev))  # 마지막 구간 추가

    return ranges


#프레임의 위치 구하기 (-> 유정)
def frame_to_timestamp(frame_name, fps):
    """
    frame_name: e.g., 'frame_0045.jpg'
    fps: frames per second (e.g., 30)
    """
    # 프레임 인덱스 추출
    basename = os.path.basename(frame_name)
    frame_num_str = basename.replace("frame_", "").replace(".jpg", "")
    frame_idx = int(frame_num_str)

    # 초로 변환
    return frame_idx / fps

#프레임 정보 구하기 (-> 유정)
def get_video_info(original_video):
    """
    ffprobe를 통해 원본 영상의 fps와 해상도(width, height)를 추출
    """
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            original_video
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        info = json.loads(result.stdout)
        stream = info["streams"][0]

        width = stream["width"]
        height = stream["height"]
        r_frame_rate = stream["r_frame_rate"]  # 예: "30/1"

        # r_frame_rate → fps로 변환
        num, denom = map(int, r_frame_rate.split("/"))
        fps = num / denom

        return fps, f"{width}x{height}"

    except Exception as e:
        print(f"[ERROR] ffprobe 실패: {e}")
        return 10, "448x256"  # 기본 fallback 값

#exp 계산
def get_max_exp_for_stride(length):
    exp = 0
    while (2 ** exp + 1) <= length:
        exp += 1
    return exp - 1 if exp > 0 else 0


#보간 하기
def conversion(video_id, original_video, frame_indices, frame_root, output_root, temp_root,
        start_time):
    try:
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(temp_root, exist_ok=True)

        fps, resolution = get_video_info(original_video)
        start_idx = frame_indices[0]
        end_idx = frame_indices[-1]

        print(f"[Interpolating]... {start_idx} to {end_idx}")

        output_filename = f"overlay_{video_id}_{start_idx:04d}_{end_idx:04d}.mp4"
        output_path = os.path.join(output_root, output_filename)

        concat_dir = os.path.join(temp_root, "concat")
        os.makedirs(concat_dir, exist_ok=True)

        # exp 자동 계산
        frame_count = len(frame_indices)
        exp = get_max_exp_for_stride(frame_count)
        interp_len = 2 ** exp + 1

        print(f"[INFO] exp={exp}, 보간 대상: {interp_len}개 프레임 중 첫/끝")

        interp_start = frame_indices[0]
        interp_end = frame_indices[2 ** exp]

        img_counter = 0

        # 보간 실행
        temp_dir = os.path.join(temp_root, "interpolation")
        os.makedirs(temp_dir, exist_ok=True)

        start_path = os.path.abspath(os.path.join(frame_root, f"frame_{interp_start:04d}.jpg"))
        end_path = os.path.abspath(os.path.join(frame_root, f"frame_{interp_end:04d}.jpg"))
        temp_dir = os.path.abspath(temp_dir)
    

        MODEL_DIR = os.getenv("MODEL_ROUTE")
        PYTHON_310 = os.path.join(MODEL_DIR, ".venv", "Scripts", "python.exe")
        INTERPOLATION = os.path.join(MODEL_DIR, "inference_img.py")

        subprocess.run([
            PYTHON_310,
            INTERPOLATION,
            "--img", start_path, end_path,
            "--exp", str(exp),
            "--out", temp_dir
        ], check=True, cwd=MODEL_DIR)
        # 보간된 이미지 전체 복사 (start~end 포함되어 있음)
        interpolated = sorted([
            f for f in os.listdir(temp_dir) if f.endswith(".jpg")
        ])
        for fname in interpolated:
            shutil.copy(os.path.join(temp_dir, fname), os.path.join(concat_dir, f"img{img_counter:04d}.jpg"))
            img_counter += 1

        # 나머지 프레임들 복사 (보간에 포함되지 않은 부분)
        for idx in frame_indices[2 ** exp + 1:]:
            remain_path = os.path.join(frame_root, f"frame_{idx:04d}.jpg")
            shutil.copy(remain_path, os.path.join(concat_dir, f"img{img_counter:04d}.jpg"))
            img_counter += 1

        # 영상 생성
        subprocess.run([
            "ffmpeg", "-y",
            "-r", str(fps),
            "-f", "image2",
            "-i", os.path.join(concat_dir, "img%04d.jpg"),
            "-s", resolution,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ], check=True)

        print(f"[SUCCESS] 보간 영상 생성 완료 → {output_path}")
        print(f"[INFO] 최종 프레임 수: {img_counter}장")

        duration = img_counter / fps
        return {
            "overlay_path": output_path,
            "start": round(start_time, 3),
            "duration": round(duration, 3)
        }

    except Exception as e:
        print(f"[ERROR] 보간 중 오류 발생: {e}")
    finally:
        if os.path.exists(temp_root):
            shutil.rmtree(temp_root)
