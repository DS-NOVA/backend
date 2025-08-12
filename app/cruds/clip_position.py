import cv2
import os
from typing import List, Dict

'''
<input>
->한 영상당 클립 안의 프레임들 png
clips = [
    ["frame_0001.png", ..., "frame_0010.png"],   # 클립 1
    ["frame_0012.png", ..., "frame_0100.png"],   # 클립 2
    ...
]

<output>
->fps, 영상의 위치(초), 프레임들
'''

# sample 파일 연결
video_path = "./backend/uploads/sample.mp4"

# FPS 계산 (초당 몇 프레임)
def get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

# output 데이터 함수
def clip_position_data(
    clips: List[List[str]],  # [["frame_0001.png", ..., "frame_0010.png"], ...]
    video_path: str
) -> Dict:
    fps = get_fps(video_path)
    clip_summaries = []

    for clip in clips:
        if not clip:
            continue  # 빈 클립은 건너뛰기

        # 시작/끝 프레임 번호 추출
        start_frame_str = os.path.splitext(clip[0])[0].split("_")[-1]
        #end_frame_str = os.path.splitext(clip[-1])[0].split("_")[-1]

        start_frame = int(start_frame_str)
        #end_frame = int(end_frame_str) + 1

        # 초 단위 시간 변환
        start_time = start_frame / fps
        #end_time = end_frame / fps

        clip_summaries.append({
            "start": round(start_time, 3), #소수점 3자리까지 반올림
            #"end": round(end_time, 3),
            "frames": clip
        })

    return {
        "fps": round(fps, 2),
        "clips": clip_summaries
    }

'''
사용예시
<input>
clips = [
    ["frame_0001.png", "frame_0002.png", "frame_0003.png"],
    ["frame_0100.png", "frame_0101.png", "frame_0102.png"]
]

result = clip_position_data(clips, video_path)

<output>
{
    "fps": 30.0,
    "clips": [
        {
        "start": 3.0,
        "end": 4.2,
        "frames": [
            "frame_090.png", "frame_091.png", ..., "frame_104.png"
        ]
        },
        {
        "start": 5.0,
        "end": 5.6,
        "frames": [
            "frame_150.png", "frame_151.png", ..., "frame_156.png"
        ]
        }
    ]
}
'''