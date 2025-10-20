from __future__ import annotations
from typing import List, Tuple, OrderedDict
from collections import OrderedDict as OD
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import cv2
import numpy as np
import os


#추후 삭제
def _mask_from_bboxes(bboxes, H, W):
    m = np.zeros((H, W), dtype=np.uint8)
    for (x,y,w,h) in bboxes:
        x0=max(0,x); y0=max(0,y); x1=min(W, x+w); y1=min(H, y+h)
        m[y0:y1, x0:x1] = 1
    return m

def _overlay_mask(frame_bgr, mask, color_bgr=(0,0,255), alpha=0.35):
    out = frame_bgr.copy()
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    overlay = out.copy()
    overlay[mask > 0] = color_bgr
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


# =========================
# 공통 유틸
# =========================
def _safe_bool_arr(length: int, init: int = 0) -> List[int]:
    return [int(bool(init))] * max(0, int(length))

def _pad_or_trim(arr: List[int], length: int) -> List[int]:
    if len(arr) < length:
        return arr + [0] * (length - len(arr))
    if len(arr) > length:
        return arr[:length]
    return arr

def _group_runs(mask: List[int]) -> List[Tuple[int,int]]:
    """연속된 1-런 (start,end)"""
    runs = []
    s = None
    for i, v in enumerate(mask):
        if v:
            if s is None: s = i
        else:
            if s is not None:
                runs.append((s, i-1))
                s = None
    if s is not None:
        runs.append((s, len(mask)-1))
    return runs

# =========================
# PHF (밝기 기반)
# =========================
def _rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    # 입력 BGR을 먼저 RGB로 바꿔서 들어오게 할 수도 있지만, 여기선 BGR→RGB 변환 후 진행
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.2126*r + 0.7152*g + 0.0722*b

def _detect_phf_pair(prev_lum: np.ndarray, curr_lum: np.ndarray,
                     delta_l_threshold=20, min_luminance_threshold=20, area_threshold=0.4) -> dict:
    delta = curr_lum - prev_lum
    res = {}
    res["프레임간_휘도차_20이상_존재"] = np.any(np.abs(delta) >= delta_l_threshold)
    res["더_어두운프레임_최소휘도_20이하"] = min(float(prev_lum.min()), float(curr_lum.min())) <= min_luminance_threshold
    res["휘도차20이상_영역_비율_40이상"] = (np.abs(delta) >= delta_l_threshold).sum() / delta.size >= area_threshold
    res["밝아진_프레임"] = float(curr_lum.mean()) - float(prev_lum.mean()) >= delta_l_threshold
    res["PHF"] = all([res["프레임간_휘도차_20이상_존재"],
                      res["더_어두운프레임_최소휘도_20이하"],
                      res["휘도차20이상_영역_비율_40이상"]])
    return res

def _expand_phf_brightening_only(luminances: List[np.ndarray], pair_results: List[dict], delta_threshold: float = 0.0) -> List[dict]:
    out = [r.copy() for r in pair_results]
    for i, r in enumerate(pair_results):
        if not r.get("PHF"): continue
        if not r.get("밝아진_프레임"): continue
        if i+1 >= len(luminances) or i+1 >= len(out): continue
        if float(luminances[i+1].mean()) - float(luminances[i].mean()) >= delta_threshold:
            out[i+1]["PHF"] = True
    return out

def compute_phf_flags(frames_bgr: List[np.ndarray]) -> List[int]:
    n = len(frames_bgr)
    if n == 0: return []
    lums = [ _rgb_to_luminance(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr ]
    pair_res = []
    for i in range(1, n):
        pair_res.append(_detect_phf_pair(lums[i-1], lums[i]))
    pair_res = _expand_phf_brightening_only(lums, pair_res, 0.0)
    flags = [0]  # 프레임0은 비교 대상 없음
    flags += [ int(bool(r.get("PHF", False))) for r in pair_res ]
    return flags

# =========================
# PHR (적색/흑색 전이)
# =========================
def _is_saturated_red(yuv: np.ndarray) -> np.ndarray:
    Y, Cr, Cb = cv2.split(yuv)
    return ((Cr > 150) & (Cb < 120)).astype(np.uint8)

def _is_black(yuv: np.ndarray) -> np.ndarray:
    Y, _, _ = cv2.split(yuv)
    return (Y < 40).astype(np.uint8)

def _expand_similar(results: List[dict], base: List[int], thr=0.05) -> List[int]:
    expanded = set(base)
    q = list(base)
    while q:
        cur = q.pop(0)
        for off in (-1, 1):
            nb = cur + off
            if 0 <= nb < len(results) and nb not in expanded:
                rdiff = abs(results[cur]['saturated_red_ratio'] - results[nb]['saturated_red_ratio'])
                bdiff = abs(results[cur]['black_ratio'] - results[nb]['black_ratio'])
                if rdiff <= thr and bdiff <= thr:
                    expanded.add(nb); q.append(nb)
    return sorted(expanded)

def compute_phr_flags(frames_bgr: List[np.ndarray]) -> List[int]:
    n = len(frames_bgr)
    if n == 0: return []
    res = []
    prev_r = prev_b = None
    for f in frames_bgr:
        yuv = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        red_ratio   = float(_is_saturated_red(yuv).sum()) / yuv.shape[0] / yuv.shape[1]
        black_ratio = float(_is_black(yuv).sum()) / yuv.shape[0] / yuv.shape[1]
        is_danger = False
        if prev_r is not None and prev_b is not None:
            red_diff = red_ratio - prev_r
            black_diff = black_ratio - prev_b
            # 조건 B: 적색 증가 + 흑색 감소
            if red_diff >= 0.25 and black_diff <= -0.25:
                is_danger = True
        res.append({
            "saturated_red_ratio": round(red_ratio, 4),
            "black_ratio": round(black_ratio, 4),
            "is_dangerous": is_danger
        })
        prev_r, prev_b = red_ratio, black_ratio
    base_idx = [i for i, r in enumerate(res) if r["is_dangerous"]]
    expanded = _expand_similar(res, base_idx, thr=0.30)
    flags = _safe_bool_arr(n, 0)
    for i in expanded: flags[i] = 1
    return flags

# =========================
# PFS (PHF 빈도 기반)
# =========================
def compute_pfs_flags(phf: List[int], fps: float) -> List[int]:
    n = len(phf)
    if n == 0: return []
    # PHF 연속 구간들
    groups = _group_runs(phf)
    # 3프레임 이하 간격으로 이어붙인 시퀀스들 중
    out = _safe_bool_arr(n, 0)
    if not groups: return out

    def flush(seq):
        nonlocal out
        if len(seq) >= 3:
            seq_s, seq_e = seq[0][0], seq[-1][1]
            dur = (seq_e - seq_s + 1) / max(1.0, fps)
            freq = len(seq) / dur if dur > 0 else 0.0
            if freq > 3.0:
                for i in range(seq_s, seq_e+1): out[i] = 1

    seq = []
    for g in groups:
        if not seq:
            seq.append(g)
        else:
            prev_end = seq[-1][1]
            if g[0] - prev_end <= 3:
                seq.append(g)
            else:
                flush(seq); seq = [g]
    flush(seq)
    return out

# =========================
# PS (지속 노출: (PHF or PHR) 유지)
# =========================
def compute_ps_flags(phf: List[int], phr: List[int], fps: float, min_duration_sec=5, max_gap_frames=5) -> List[int]:
    n = min(len(phf), len(phr))
    if n == 0: return []
    is_flash = [1 if (phf[i] or phr[i]) else 0 for i in range(n)]
    out = _safe_bool_arr(n, 0)
    start = end = None; gap = 0
    min_len = int(round(min_duration_sec * fps))
    for i, v in enumerate(is_flash):
        if v:
            if start is None: start = i
            end = i; gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > max_gap_frames:
                    if end is not None and (end - start + 1) >= min_len:
                        for k in range(start, end+1): out[k] = 1
                    start = end = None; gap = 0
    if start is not None and end is not None and (end - start + 1) >= min_len:
        for k in range(start, end+1): out[k] = 1
    return out

# =========================
# PHP / PSP (줄무늬 + 합집합 면적)
# — 저장 없이 핵심만 수행 (가보르, 연결요소, stripe pair 카운트)
# =========================
@dataclass
class PSEConfig:
    gabor_orientations_deg: Tuple[int, ...] = (0, 45, 90, 135)
    gabor_wavelengths_px: Tuple[int, ...] = (6, 8, 12, 16)
    scale_factor: float = 0.75
    heavy_every_n: int = 1
    gabor_energy_thresh_percentile: float = 98.0 #이 수치를 올리면 과탐 억제
    min_region_area_frac: float = 0.045 #이 수치를 올리면 과탐 억제
    min_stripe_pairs: int = 5 #이 수치는 건들면 안됨됨
    psp_area_frac_threshold: float = 0.40
    bbox_shrink_frac: float = 0.2
    orientation_coherence_thresh: float = 0.78

def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(u <= 0.04045, u / 12.92, ((u + a) / (1 + a)) ** 2.4)

def _relative_luminance_from_bgr(bgr: np.ndarray) -> np.ndarray:
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    rgb_lin = _srgb_to_linear(rgb)
    Y = 0.2126*rgb_lin[...,0] + 0.7152*rgb_lin[...,1] + 0.0722*rgb_lin[...,2]
    return Y.astype(np.float32)

class _GaborBank:
    def __init__(self, orientations: Tuple[int,...], wavelengths: Tuple[int,...]):
        self.oris = orientations; self.lambs = tuple(int(max(4, round(l))) for l in wavelengths)
        self.kernels = {}
        self._build()
    def _build(self):
        for th in self.oris:
            theta = np.deg2rad(th)
            for lamb in self.lambs:
                sigma = 0.56 * lamb; gamma = 0.5; psi = 0
                ksize = int(max(7, round(6*sigma))); ksize += (ksize % 2 == 0)
                self.kernels[(th, lamb)] = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamb, gamma, psi, ktype=cv2.CV_32F)
    def energy_and_orientation(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        energies = []
        for th in self.oris:
            esum = np.zeros_like(Y, dtype=np.float32)
            for lamb in self.lambs:
                k = self.kernels[(th,lamb)]
                resp = cv2.filter2D(Y, cv2.CV_32F, k)
                esum += cv2.magnitude(resp, np.zeros_like(resp))
            energies.append(esum)
        E = np.stack(energies, axis=-1)  # (H,W,O)
        maxE = E.max(axis=-1)
        ori_idx = E.argmax(axis=-1).astype(np.int32)
        return maxE, ori_idx

def _count_stripe_pairs(Y: np.ndarray, bbox: Tuple[int,int,int,int], dominant_theta_deg: float) -> int:
    x,y,w,h = bbox
    roi = Y[y:y+h, x:x+w]
    if roi.size == 0: return 0
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), dominant_theta_deg, 1.0)
    rotated = cv2.warpAffine(roi, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    profile = rotated.mean(axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (1,0), sigmaX=1.0).ravel()
    # 간단한 피크/계곡 카운트(강도 중간 이상만)
    d = np.diff(profile); s = np.sign(d); s[s==0] = 1e-6
    z = np.where(np.diff(np.sign(s)) != 0)[0]
    if len(z) < 2: return 0
    vals = profile[z+1]
    amp = np.abs(vals - profile.mean())
    if amp.max() < 1e-3: return 0
    thr = np.quantile(amp, 0.5)
    good = (amp >= thr).sum()
    return int(good // 2)


def compute_php_psp_flags(
    frames_bgr: List[np.ndarray],
    cfg: PSEConfig = PSEConfig(),
    debug_save_dir: Path | None = None,      # 디버그 PNG 저장 폴더(옵션)
    debug_draw: str = "both",                # "php" | "psp" | "both"
    save_bbox_outline: bool = False          # 박스 외곽선
) -> Tuple[List[int], List[int]]:
    n = len(frames_bgr)
    if n == 0: return [], []

    php = _safe_bool_arr(n, 0)
    psp = _safe_bool_arr(n, 0)

    # 원본과 동일: 작은 스케일 가보르로 후보 찾고 방향/에너지 계산
    gb_small = _GaborBank(
        cfg.gabor_orientations_deg,
        tuple(max(4, int(round(w * cfg.scale_factor))) for w in cfg.gabor_wavelengths_px)
    )

    H, W = frames_bgr[0].shape[:2]
    min_area = cfg.min_region_area_frac * H * W
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  # 원본: (5,5)

    for i, f in enumerate(frames_bgr):
        heavy = (cfg.heavy_every_n <= 1) or (i % cfg.heavy_every_n == 0)

        has_php = False
        has_psp = False
        union_mask = np.zeros((H, W), dtype=np.uint8)
        bboxes = []

        if heavy:
            Y = _relative_luminance_from_bgr(f)
            Y_small = cv2.resize(Y, None, fx=cfg.scale_factor, fy=cfg.scale_factor, interpolation=cv2.INTER_AREA)

            maxE_small, ori_idx_small = gb_small.energy_and_orientation(Y_small)
            thr = np.percentile(maxE_small, cfg.gabor_energy_thresh_percentile)
            mask_small = (maxE_small >= thr).astype(np.uint8)

            # 원본 형태학 복원: CLOSE → DILATE
            mask_up = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            mask_up = cv2.morphologyEx(mask_up, cv2.MORPH_CLOSE, kern, iterations=1)
            mask_up = cv2.dilate(mask_up, kern, iterations=1)

            # 방향도 원본처럼 업샘플
            ori_idx = cv2.resize(ori_idx_small, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_up, connectivity=8)

            for lab in range(1, num_labels):
                x, y, w, h, area = stats[lab]
                if area < min_area:
                    continue

                region_mask = (labels == lab)
                if not region_mask.any():
                    continue

                ori_in = ori_idx[region_mask]
                if ori_in.size == 0:
                    continue

                ori_id = int(np.bincount(ori_in.ravel()).argmax())
                theta = cfg.gabor_orientations_deg[ori_id]

                # shrink 제거(원본대로)
                pairs = _count_stripe_pairs(Y, (int(x), int(y), int(w), int(h)), float(theta))
                if pairs >= cfg.min_stripe_pairs:
                    bx, by, bw, bh = int(x), int(y), int(w), int(h)
                    bboxes.append((bx, by, bw, bh))

            has_php = len(bboxes) > 0

            if has_php:
                # PSP: bbox 합집합 면적 비율
                for (x, y, w, h) in bboxes:
                    x0 = max(0, x); y0 = max(0, y); x1 = min(W, x + w); y1 = min(H, y + h)
                    union_mask[y0:y1, x0:x1] = 1
                union_area_frac = float(union_mask.sum()) / float(H * W)
                has_psp = (union_area_frac >= cfg.psp_area_frac_threshold)

            # === 디버그 오버레이 저장 (원함 유지) ===
            if debug_save_dir is not None:
                debug_save_dir.mkdir(parents=True, exist_ok=True)
                out_img = f.copy()
                if debug_draw in ("php", "both") and has_php:
                    php_mask = _mask_from_bboxes(bboxes, H, W)
                    out_img = _overlay_mask(out_img, php_mask, color_bgr=(0,255,255), alpha=0.35)  # 노랑
                if debug_draw in ("psp", "both") and has_psp:
                    out_img = _overlay_mask(out_img, union_mask, color_bgr=(0,0,255), alpha=0.35)  # 빨강
                if save_bbox_outline and has_php:
                    for (x, y, w, h) in bboxes:
                        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0,255,255), 1, cv2.LINE_AA)
                cv2.imwrite(str(Path(debug_save_dir) / f"frame_{i:06d}.png"), out_img)

        php[i] = int(has_php)
        psp[i] = int(has_psp)

    return php, psp

# =========================
# 메인 실행기
# =========================
def run_cv_metrics(video_path: Path, fps: float | None = None,
                   debug_save_dir: Path | None = None, debug_draw: str = "both",
                   save_bbox_outline: bool = False) -> "OrderedDict[str, List[int]]":

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return OD()
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = fps if (fps and fps > 0) else (cap_fps if cap_fps > 0 else 30.0)

    phf=[]; phr=[]; php=[]; psp=[]
    prev_bgr = None
    prev_lum = None
    pse_cfg = PSEConfig(
        heavy_every_n=2,                      # 가보르 빈도 ↓
        scale_factor=0.65,                    # 축소 ↑
        gabor_energy_thresh_percentile=99.0,  # 과탐 억제
        min_region_area_frac=0.06
    )

    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break

        # 미리 계산해 재사용
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lum = _rgb_to_luminance(rgb)  # float32

        if prev_bgr is None:
            phf.append(0); phr.append(0); php.append(0); psp.append(0)
        else:
            # PHF(한 번 계산한 휘도 재사용)
            r = _detect_phf_pair(prev_lum, lum)
            phf.append(int(bool(r["PHF"])))

            # PHR(현재 프레임 단독 통계 + 전 프레임 대비)
            yuv_prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2YCrCb)
            yuv_curr = cv2.cvtColor(bgr,      cv2.COLOR_BGR2YCrCb)
            # 포뮬라는 기존 compute_phr_flags와 동일하게 구현
            red_prev   = float((_is_saturated_red(yuv_prev)).sum()) / yuv_prev.shape[0] / yuv_prev.shape[1]
            black_prev = float((_is_black(yuv_prev)).sum())         / yuv_prev.shape[0] / yuv_prev.shape[1]
            red_curr   = float((_is_saturated_red(yuv_curr)).sum()) / yuv_curr.shape[0] / yuv_curr.shape[1]
            black_curr = float((_is_black(yuv_curr)).sum())         / yuv_curr.shape[0] / yuv_curr.shape[1]
            is_danger = (red_curr - red_prev >= 0.25) and (black_curr - black_prev <= -0.25)
            phr.append(int(is_danger))

            # PHP/PSP: 최근 몇 프레임 내 PHF/PHR 중 하나라도 1이면만 수행
            need_heavy = any(phf[max(0,i-7):i]) or any(phr[max(0,i-7):i])
            if need_heavy:
                _php, _psp = compute_php_psp_flags([bgr], pse_cfg, None, "both", False)
                php.append(_php[0]); psp.append(_psp[0])
            else:
                php.append(0); psp.append(0)

        prev_bgr = bgr
        prev_lum = lum
        i += 1

    cap.release()

    # 파생 플래그
    pfs = compute_pfs_flags(phf, fps)
    ps  = compute_ps_flags(phf, phr, fps, min_duration_sec=5, max_gap_frames=5)

    n = len(phf)
    return OD([
        ("phf", _pad_or_trim(phf, n)),
        ("phr", _pad_or_trim(phr, n)),
        ("pfs", _pad_or_trim(pfs, n)),
        ("ps",  _pad_or_trim(ps,  n)),
        ("php", _pad_or_trim(php, n)),
        ("psp", _pad_or_trim(psp, n)),
    ])


def dump_json(metrics: "OrderedDict[str, List[int]]", out_path: Path, fps: float | None = None) -> Path:
    out = {
        "labels": list(metrics.keys()),
        "num_frames": len(next(iter(metrics.values()))) if metrics else 0,
        "fps": fps,
        "flags": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
