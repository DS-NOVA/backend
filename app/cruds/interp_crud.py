import torch
import shutil
from pathlib import Path
from PIL import Image
import sys
import os
from typing import List, Tuple
import imageio.v2 as imageio
import cv2
import numpy as np
import math
from dataclasses import dataclass


# 모델 임포트
sys.path.append(os.path.abspath("C:/nova/IFRNet-main"))
from models.IFRNet import Model

# cpu 사용
import torch
from pathlib import Path

_IFR_MODEL = None
_IFR_DEVICE = torch.device("cpu")  

def get_ifrnet(checkpoint_path: Path):
    global _IFR_MODEL
    if _IFR_MODEL is None:
        m = Model().to(_IFR_DEVICE)
        sd = torch.load(str(checkpoint_path), map_location=_IFR_DEVICE) 
        m.load_state_dict(sd)
        m.eval()
        _IFR_MODEL = m
    return _IFR_MODEL

# 위험 구간 묶기
def group_consecutive_ones(mask: List[int], min_len: int = 2) -> List[Tuple[int, int]]:
    res = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i] == 1:
            j = i
            while j + 1 < n and mask[j + 1] == 1:
                j += 1
            if (j - i + 1) >= min_len:
                res.append((i, j))
            i = j + 1
        else:
            i += 1
    return res

# 밝기 조절
def adjust_brightness_masked_pil(
        img_pil: Image.Image, 
        thresh: int = 140, 
        atten_factor: float = 0.75, 
        blur_ksize: int = 31, 
        print_coverage: bool = False
) -> Image.Image:
    rgb = np.array(img_pil) 
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    if blur_ksize and blur_ksize > 1 and blur_ksize % 2 == 1:
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    if print_coverage:
        coverage = float((mask > 0).sum()) / mask.size
        print(f"[BRIGHT-MASK] coverage={coverage:.2%}, thresh={thresh}, atten={atten_factor}, blur={blur_ksize}")

    mask_norm = (mask.astype(np.float32) / 255.0)[..., None]  
    rgb_f = rgb.astype(np.float32)

    out = rgb_f * (1 - mask_norm + mask_norm * atten_factor)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

# 밝기 평균 유틸
def frame_mean_luma_png(p: Path) -> float:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())

# risky run 밝기 평균 임계값
def build_adaptive_thresh_by_runs(
        frame_paths: list[Path],
        risky_ranges: list[tuple[int,int]],
        margin: int = 0,
        clamp_low: int = 60,
        clamp_high: int = 245
) -> dict[int, int]:
    per_run_thresh: dict[int,int] = {}
    for run_idx, (s, _e) in enumerate(risky_ranges):
        if 0 <= s < len(frame_paths):
            mu = frame_mean_luma_png(frame_paths[s])  # 0~255
            thr = int(round(mu + margin))
            thr = max(clamp_low, min(clamp_high, thr))
            per_run_thresh[run_idx] = thr
        else:
            # fallback (안전)
            per_run_thresh[run_idx] = 140
    return per_run_thresh

# 각 프레임의 "어느 런(run)에 속하는지", "런에서 몇 번째인지", "런 길이"를 구함
def build_run_index(total_frames: int, risky_ranges: list[tuple[int,int]]):
    frame_to_run = [None] * total_frames # 각 프레임의 런 인덱스
    idx_in_run = [None] * total_frames # 그 런에서의 상대 인덱스(0..len-1)
    run_len = [None] * total_frames # 그 런의 총 길이
    for ridx, (s, e) in enumerate(risky_ranges or []):
        s = max(0, s); e = min(total_frames - 1, e)
        L = max(1, e - s + 1)
        for i in range(s, e+1):
            frame_to_run[i] = ridx
            idx_in_run[i]   = i - s
            run_len[i]      = L
    return frame_to_run, idx_in_run, run_len

#램프 보정 (추후 수정)
def _hann_envelope(idx: float, n_frames: int, hold_ratio: float = 0.0) -> float:
    if n_frames <= 1:
        return 0.0
    t = max(0.0, min(1.0, idx / max(1.0, (n_frames - 1))))
    hold_ratio = max(0.0, min(0.8, hold_ratio))
    if hold_ratio == 0.0:
        return float(math.sin(math.pi * t) ** 2)

    a = 0.5 * (1.0 - hold_ratio)
    if t < a:
        x = t / a
        return float(math.sin(math.pi * x / 2) ** 2)
    if t > 1.0 - a:
        x = (t - (1.0 - a)) / a
        return float(math.cos(math.pi * x / 2) ** 2)
    return 1.0

# 밝기 및 채도 조절에 사용할 데이터 
@dataclass
class RampVSParams:
    mode: str = "both"     # "brightness" | "saturation" | "both"
    v_min: float = 0.75    # 0..1 (밝기 최소 배율)
    s_min: float = 0.75    # 0..1 (채도 최소 배율)
    gamma_max: float = 1.00 # >=1 (1이면 감마 무효)
    hold_ratio: float = 0.05 # 0..0.8

#밝기와 채도 적용하기 
def apply_vs_ramp_to_rgb_linear(rgb: np.ndarray, s: float, p: RampVSParams) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    x_lin = np.power(np.clip(x, 0.0, 1.0), 2.2)

    #밝기, 채도 스케일 계산
    b = 1.0 - s * (1.0 - p.v_min) # brightness
    sat = 1.0 - s * (1.0 - p.s_min) # saturation

    #채도: y + (x - y) * sat   (y는 휘도)
    wr, wg, wb = 0.2126, 0.7152, 0.0722
    y = (x_lin[...,0]*wr + x_lin[...,1]*wg + x_lin[...,2]*wb)[..., None]
    x_sat = y + (x_lin - y) * sat

    #밝기: 전체 스케일
    out_lin = np.clip(x_sat * b, 0.0, 1.0)

    #감마
    out = (np.power(out_lin, 1.0/2.2) * 255.0 + 0.5).astype(np.uint8)
    return out


def map_frame_to_run(total_frames: int, risky_ranges: list[tuple[int,int]]) -> list[int | None]:
    run_of = [None] * total_frames
    for run_idx, (s, e) in enumerate(risky_ranges):
        s = max(0, s); e = min(total_frames-1, e)
        for i in range(s, e+1):
            run_of[i] = run_idx
    return run_of

def merge_ones_with_gap(mask: List[int], max_gap: int = 3, min_len: int = 2) -> List[Tuple[int, int]]:
    """
    0/1 시퀀스에서 '1'들이 서로 max_gap 이하 간격(0의 개수)으로 등장하면
    하나의 위험 시퀀스로 묶어 (start_idx, end_idx)로 반환.
    - min_len: 최종 묶인 구간의 길이(끝-시작+1)가 이 값 미만이면 제외.
    예) 1,1,0,0,1  (max_gap=3)  -> (0,4)
        1,0,0,0,0,1 (gap=4>3) -> (0,0), (5,5) 이지만 둘 다 싱글이라 제외
    """
    n = len(mask)
    res: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        # 첫 1 찾기
        while i < n and mask[i] == 0:
            i += 1
        if i >= n:
            break
        start = i
        ones_cnt = 1
        j = i
        while j + 1 < n:
            if mask[j + 1] == 1:
                j += 1
                ones_cnt += 1
            else:
                # j+1부터 0이 몇 개 이어지는지 보고, 그 뒤에 1이 나오면 브릿지
                k = j + 1
                zero_cnt = 0
                while k < n and mask[k] == 0 and zero_cnt < max_gap:
                    zero_cnt += 1
                    k += 1
                # 0이 1~max_gap개였고 그 뒤에 1을 만나면 이어 붙임
                if k < n and mask[k] == 1 and 1 <= zero_cnt <= max_gap:
                    j = k
                    ones_cnt += 1
                else:
                    break
        end = j
        if (end - start + 1) >= min_len and ones_cnt >= 2:
            res.append((start, end))
        i = end + 1
    return res

def _temporal_smooth(arr: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    out = arr.copy()
    # forward
    for i in range(1, len(out)):
        out[i] = alpha * out[i] + (1 - alpha) * out[i - 1]
    # backward
    for i in range(len(out) - 2, -1, -1):
        out[i] = alpha * out[i] + (1 - alpha) * out[i + 1]
    return out


def assemble_interpolated_video(
        video_id: str,
        fps: float,
        base_dir: str,
        frame_ext: str = ".png",
        checkpoint_abs_path: str | None = None,
        insert_mode: str = "risky_only",
        risky_ranges: List[Tuple[int,int]] | None = None,
        risky_mask: List[int] | None = None,
        use_vs_ramp: bool = True,
        ramp_params: RampVSParams = RampVSParams(
            mode="both", v_min=0.75, s_min=0.75, gamma_max=1.0, hold_ratio=0.05
        ),
        save_intermediate: bool = True,
        dump_dir_name: str = "risky_dump",
        overwrite_dump: bool = True,
        stream_from_raw: bool = True,
        raw_video_path: Path | None = None
):
    print(f"[DEBUG][assemble] ramp_params: v_min={ramp_params.v_min}, s_min={ramp_params.s_min}, gamma_max={ramp_params.gamma_max}, hold_ratio={ramp_params.hold_ratio}")

    PROJECT_ROOT = Path(base_dir)
    STATIC_DIR   = PROJECT_ROOT / "static"
    video_dir    = STATIC_DIR / "uploads" / video_id
    frames_dir   = video_dir / "frames"
    overlays_dir = video_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    dump_dir = overlays_dir / dump_dir_name
    if save_intermediate:
        if overwrite_dump and dump_dir.exists():
            shutil.rmtree(dump_dir)
        (dump_dir / "originals").mkdir(parents=True, exist_ok=True)
        (dump_dir / "mids").mkdir(parents=True, exist_ok=True)

    # 체크포인트/모델 (CPU 고정, 1회 캐시)
    checkpoint_path = Path(checkpoint_abs_path) if checkpoint_abs_path else PROJECT_ROOT / "IFRNet-main" / "IFRNet_Vimeo90K.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 파일이 없습니다: {checkpoint_path}")

    model = get_ifrnet(checkpoint_path)
    model_device = next(model.parameters()).device  # => cpu

    # 총 프레임 수/해상도는 frames_dir를 그대로 사용 (현재 라우터에서 extract_frames를 돌리고 있으므로 존재)
    frame_paths = sorted(frames_dir.glob(f"frame_*{frame_ext}"))
    if not frame_paths:
        raise FileNotFoundError(f"프레임이 없습니다: {frames_dir}")
    total = len(frame_paths)
    base_w, base_h = Image.open(frame_paths[0]).size  # (필요 시 사용)

    # 위험 런 인덱스/램프값 사전 계산
    frame_to_run, idx_in_run, run_len = build_run_index(total, risky_ranges or [])
    s_val = np.zeros(total, dtype=np.float32)
    edge_ratio = 0.15  # 구간 길이의 15%를 입·출구 램프에 할당 (0.10~0.20 권장)
    for ridx, (s, e) in enumerate(risky_ranges or []):
        L = max(1, e - s + 1)
        env = np.ones(L, dtype=np.float32)
        edge = int(max(1, round(L * edge_ratio)))
        # 입구 램프: 0→1, 출구 램프: 1→0 (부드러운 코사인 제곱)
        t = np.linspace(0.0, np.pi/2, edge, dtype=np.float32)
        in_ramp  = (np.sin(t))**2
        out_ramp = in_ramp[::-1]
        if edge < L:
            env[:edge]  = in_ramp
            env[-edge:] = out_ramp
            env[edge:L-edge] = 1.0  # 중앙 평탄
        else:
            # 구간이 매우 짧으면 그냥 sin^2로
            tt = np.linspace(0.0, 1.0, L, dtype=np.float32)
            env = (np.sin(np.pi*tt))**2
        s_val[s:e+1] = env

    s_smooth = _temporal_smooth(s_val, alpha=0.15)

    # mid 삽입 위치 결정
    insert_mid = [False] * max(0, total - 1)
    if insert_mode == "risky_only":
        if risky_ranges:
            for (s, e) in risky_ranges:
                for k in range(s, e):
                    if 0 <= k < total - 1:
                        insert_mid[k] = True
    elif insert_mode == "mask":
        if risky_mask is None:
            raise ValueError("insert_mode='mask' requires risky_mask")
        for k in range(min(len(risky_mask), total - 1)):
            insert_mid[k] = bool(risky_mask[k])
    else:
        raise ValueError("insert_mode must be 'risky_only' or 'mask'")

    out_path = overlays_dir / f"{video_id}_interp_full.mp4"
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        macro_block_size=1,
        output_params=[
            "-preset", "veryfast",
            "-crf", "23",
            "-threads", "0",
            "-pix_fmt", "yuv420p",  
            "-vsync", "cfr"          
        ]
    )

    if not (stream_from_raw and raw_video_path and raw_video_path.exists()):
        raise RuntimeError("stream_from_raw=True 인데 raw_video_path가 없거나 접근 불가합니다.")

    cap = cv2.VideoCapture(str(raw_video_path))
    if not cap.isOpened():
        writer.close()
        raise RuntimeError(f"raw read fail: {raw_video_path}")

    try:
        ok, bgr_prev = cap.read()
        if not ok:
            raise RuntimeError("raw read fail (first frame)")
        target_h, target_w = bgr_prev.shape[:2]
        k = 0
        while True:
            if k >= total:
                break
            # BGR → RGB (1회)
            rgb_prev = cv2.cvtColor(bgr_prev, cv2.COLOR_BGR2RGB)
            if rgb_prev.shape[:2] != (target_h, target_w):
                rgb_prev = cv2.resize(rgb_prev, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # 램프 적용
            if use_vs_ramp and s_smooth[k] > 1e-6:
                rgb_prev = apply_vs_ramp_to_rgb_linear(rgb_prev, float(s_smooth[k]), ramp_params)

            if save_intermediate and frame_to_run[k] is not None:
                Image.fromarray(rgb_prev).save(dump_dir / "originals" / f"frame_{k:04d}.png")

            writer.append_data(rgb_prev)

            ok, bgr_next = cap.read()
            if not ok:
                break  # 마지막 프레임은 위에서 이미 append

            if insert_mid[k]:
                rgb_next = cv2.cvtColor(bgr_next, cv2.COLOR_BGR2RGB)
                if rgb_next.shape[:2] != (target_h, target_w):
                    rgb_next = cv2.resize(rgb_next, (target_w, target_h), interpolation=cv2.INTER_AREA)
                if use_vs_ramp and s_smooth[k+1] > 1e-6:
                    rgb_next = apply_vs_ramp_to_rgb_linear(rgb_next, float(s_smooth[k+1]), ramp_params)

                # IFRNet 보간
                t1 = torch.from_numpy(rgb_prev).permute(2,0,1).float().div_(255.0).unsqueeze(0).to(model_device)
                t2 = torch.from_numpy(rgb_next).permute(2,0,1).float().div_(255.0).unsqueeze(0).to(model_device)
                t  = torch.tensor([0.5], dtype=torch.float32, device=model_device).view(1,1,1,1)

                assert t1.device == next(model.parameters()).device

                with torch.inference_mode():
                    It = model.inference(t1, t2, t)

                mid_rgb = (It.squeeze(0).clamp(0,1).mul_(255).byte().permute(1,2,0).cpu().numpy())
                if mid_rgb.shape[:2] != (target_h, target_w):
                    mid_rgb = cv2.resize(mid_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                # mid 램프(평균 s값)
                if use_vs_ramp:
                    mid_s = float(0.5 * (s_smooth[k] + s_smooth[k+1]))
                    if mid_s > 1e-6:
                        mid_rgb = apply_vs_ramp_to_rgb_linear(mid_rgb, mid_s, ramp_params)

                if save_intermediate and frame_to_run[k] is not None:
                    Image.fromarray(mid_rgb).save(dump_dir / "mids" / f"mid_{k:04d}.png")

                writer.append_data(mid_rgb)

            bgr_prev = bgr_next
            k += 1
    finally:
        cap.release()
        writer.close()

    return {"output_path": str(out_path), "inserted": int(sum(insert_mid)), "dump_dir": str(dump_dir) if save_intermediate else None}
