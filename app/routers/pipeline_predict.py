from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import timm
import joblib


router = APIRouter(prefix="/nova/dashboard/video/upload")


PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
MODEL_PATH   = PROJECT_ROOT / "models" / "saved_models" / "model_lightgbm.pkl"
LABEL_NAMES  = PROJECT_ROOT / "models" / "saved_models" / "label_names.npy"

CLIP_LENGTH  = 16
FRAME_SIZE   = (224, 224)   
EMBED_DIM    = 768


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
model.to(device).eval()

_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def _extract_feature_from_clip(clip_np: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(clip_np.astype(np.float32) / 255.0).permute(0,3,1,2)
    x = (x - _mean) / _std
    feats = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            out = model.forward_features(x[i:i+1].to(device))
            cls = out[:,0,:].squeeze(0).cpu().numpy()
            feats.append(cls)
    return np.stack(feats, axis=0)  


def _make_clips_in_memory(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    clips  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) == CLIP_LENGTH:
            clips.append(np.stack(frames, axis=0))
            frames.clear()

    # 남은 프레임 버림
    if 0 < len(frames) < CLIP_LENGTH:
        pass

    cap.release()
    return clips, total_frames

def _predict_flags_and_probs(model_clf, X_df_or_np, label_names, multilabel=False, thresh=0.5):
    if hasattr(model_clf, "predict_proba"):
        proba_raw = model_clf.predict_proba(X_df_or_np)
    else:
        proba_raw = None

    # multiclass
    if isinstance(proba_raw, np.ndarray):
        flags = np.argmax(proba_raw, axis=1)
        name_to_idx = {n: i for i, n in enumerate(label_names)}
        probs_dict = {
            'flash':   (proba_raw[:, name_to_idx['flash']]   if 'flash'   in name_to_idx else np.zeros(proba_raw.shape[0])),
            'pattern': (proba_raw[:, name_to_idx['pattern']] if 'pattern' in name_to_idx else np.zeros(proba_raw.shape[0])),
            'redlight':(proba_raw[:, name_to_idx['redlight']]if 'redlight'in name_to_idx else np.zeros(proba_raw.shape[0])),
        }
        return 'multiclass', flags, probs_dict

    elif isinstance(proba_raw, list):
        pos = [ (p[:,1] if (isinstance(p, np.ndarray) and p.ndim==2 and p.shape[1]>=2) else np.zeros(X_df_or_np.shape[0]))
                for p in proba_raw ]
        pos = np.vstack(pos).T  # (N, L)
        flags = (pos >= thresh).astype(int)
        name_to_idx = {n: i for i, n in enumerate(label_names)}
        probs_dict = {}
        for t in ['flash','pattern','redlight']:
            i = name_to_idx.get(t, None)
            probs_dict[t] = (pos[:,i] if i is not None and i < pos.shape[1] else np.zeros(X_df_or_np.shape[0]))
        return 'multilabel', flags, probs_dict

    else:
        y_pred = np.asarray(model_clf.predict(X_df_or_np))
        if y_pred.ndim == 1:
            probs_dict = {k: np.zeros_like(y_pred, dtype=float) for k in ['flash','pattern','redlight']}
            return 'multiclass', y_pred, probs_dict
        else:
            flags = (y_pred > 0.5).astype(int) if y_pred.dtype.kind == 'f' else y_pred.astype(int)
            probs_dict = {k: np.zeros(flags.shape[0], dtype=float) for k in ['flash','pattern','redlight']}
            return 'multilabel', flags, probs_dict

class PredictRequest(BaseModel):
    filename: str

@router.post("/")
def predict(req: PredictRequest):
    """
    파이프라인 스크립트에서 하던 것:
      1) 영상 → 16프레임 클립들
      2) 각 클립 (16,768) 임베딩 추출 후 평균 → X (N_clips,768)
      3) 같은 X에 대해 predict / predict_proba 둘 다 호출
      4) 각 클립 결과를 프레임 단위로 전개하여 JSON 반환
    """
    video_path = RAW_DIR / req.filename
    clips, total_frames = _make_clips_in_memory(video_path)
    if clips is None:
        return {"status": "failed", "reason": "Cannot open video"}

    clip_means = []
    for clip in clips:
        feats = _extract_feature_from_clip(clip) 
        if feats.shape != (CLIP_LENGTH, EMBED_DIM):
            return {"status": "failed", "reason": f"Feature shape mismatch: {feats.shape}"}
        clip_means.append(feats.mean(axis=0))
    if not clip_means:
        return {"status": "failed", "reason": "No features extracted"}

    X_clipmean = np.stack(clip_means, axis=0)  
    model_clf = joblib.load(MODEL_PATH)
    feature_names = getattr(model_clf, "feature_name_", None)
    if feature_names is not None and len(feature_names) == X_clipmean.shape[1]:
        X_infer = pd.DataFrame(X_clipmean, columns=list(feature_names))
    else:
        X_infer = X_clipmean

    label_names = np.load(LABEL_NAMES, allow_pickle=True).tolist()
    mode, flags_raw, probs_dict = _predict_flags_and_probs(model_clf, X_infer, label_names, multilabel=False, thresh=0.5)

    n_clips = X_infer.shape[0]
    total_frames = len(clips) * CLIP_LENGTH

    flash_flags   = [0]*total_frames
    pattern_flags = [0]*total_frames
    redlight_flags= [0]*total_frames
    flash_probs   = [0.0]*total_frames
    pattern_probs = [0.0]*total_frames
    redlight_probs= [0.0]*total_frames

    # label → index
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    i_flash   = name_to_idx.get('flash', None)
    i_pattern = name_to_idx.get('pattern', None)
    i_red     = name_to_idx.get('redlight', None)

    for ci in range(n_clips):
        s = ci * CLIP_LENGTH
        e = s + CLIP_LENGTH

        # probs 전개
        if 'flash' in probs_dict and len(probs_dict['flash']) == n_clips:
            p = float(probs_dict['flash'][ci]);   flash_probs[s:e]   = [p]*CLIP_LENGTH
        if 'pattern' in probs_dict and len(probs_dict['pattern']) == n_clips:
            p = float(probs_dict['pattern'][ci]); pattern_probs[s:e] = [p]*CLIP_LENGTH
        if 'redlight' in probs_dict and len(probs_dict['redlight']) == n_clips:
            p = float(probs_dict['redlight'][ci]); redlight_probs[s:e]= [p]*CLIP_LENGTH

        # flags 전개
        if isinstance(flags_raw, np.ndarray) and flags_raw.ndim == 1:
            k = int(flags_raw[ci])
            if i_flash   is not None and k == i_flash:     flash_flags[s:e]   = [1]*CLIP_LENGTH
            if i_pattern is not None and k == i_pattern:   pattern_flags[s:e] = [1]*CLIP_LENGTH
            if i_red     is not None and k == i_red:       redlight_flags[s:e]= [1]*CLIP_LENGTH
        else:
            if i_flash   is not None and int(flags_raw[ci, i_flash])   == 1: flash_flags[s:e]   = [1]*CLIP_LENGTH
            if i_pattern is not None and int(flags_raw[ci, i_pattern]) == 1: pattern_flags[s:e] = [1]*CLIP_LENGTH
            if i_red     is not None and int(flags_raw[ci, i_red])     == 1: redlight_flags[s:e]= [1]*CLIP_LENGTH

    return {
        "status": "saved",
        "result": {
            "video": req.filename,
            "num_frames": total_frames,
            "flags": {
                "flash":   flash_flags,
                "pattern": pattern_flags,
                "redlight": redlight_flags
            },
            "probs": {
                "flash":   flash_probs,
                "pattern": pattern_probs,
                "redlight": redlight_probs
            }
        }
    }



def predict_pipeline(filename: str) -> dict:
    return predict(PredictRequest(filename=filename))