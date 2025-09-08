from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel

from app.db.database import get_db
from app.schemas.history import HistoryResponse
from app.schemas.history import HistoryListResponse, HistoryDeleteResponse
from app.models.history import History
from app.security.auth import get_current_user
from app.cruds.history_crud import delete_history_by_video
from app.models.video import Video
from app.routers.upload_router import STATIC_DIR
from app.db.mongo import get_results_coll

router = APIRouter(
    prefix="/nova/history",
    tags=["History"]
)

# 조회 API
@router.get("/list", response_model=HistoryListResponse)  # 모델에 total/limit/offset 추가 필요
def list_history(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    subq = (
        db.query(
            History.video_id.label("video_id"),
            func.max(History.viewed_at).label("last_viewed_at"),
        )
        .filter(History.user_id == user.id)
        .group_by(History.video_id)
        .subquery()
    )

    # 총 개수 (video_id의 distinct 개수)
    total = db.query(func.count()).select_from(subq).scalar() or 0

    rows = (
        db.query(
            Video.id.label("video_id"),
            Video.video_name.label("video_title"),
            Video.upload_date.label("upload_date"),
            subq.c.last_viewed_at,
        )
        .join(subq, subq.c.video_id == Video.id)
        .order_by(subq.c.last_viewed_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "history": [
            HistoryResponse(
                video_id=r.video_id,
                video_title=r.video_title,
                upload_date=r.upload_date,
            )
            for r in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }

# 삭제 API
# app/routers/history_router.py 일부
def _safe_delete_dir(dirpath: Path, base: Path) -> bool:
    try:
        dirpath = dirpath.resolve()
        base = base.resolve()
        if not str(dirpath).startswith(str(base)):
            print(f"[WARN] directory escape blocked: {dirpath}")
            return False
        if dirpath.exists():
            shutil.rmtree(dirpath)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] rmtree failed: {e}")
        return False

@router.delete("/{video_id}", response_model=HistoryDeleteResponse)
async def remove_history(
    video_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    # 1) 소유/존재 확인
    video = db.query(Video).filter_by(id=video_id, user_id=user.id).first()
    if not video:
        raise HTTPException(status_code=404, detail="해당 영상이 없습니다.")

    # 삭제에 필요한 값 미리 보관(커밋 후 객체 접근 안전)
    ek = (video.external_key or "").strip()

    # 2) 히스토리 삭제
    history_deleted = delete_history_by_video(db, video_id) or 0

    # 3) Video 삭제
    db.delete(video)
    db.commit()  # RDB 먼저 확정

    # 4) Mongo 삭제
    mongo_deleted = 0
    try:
        coll = get_results_coll()
        res = await coll.delete_one({"video_id": int(video_id)})
        mongo_deleted = getattr(res, "deleted_count", 0) or 0
    except Exception as e:
        print(f"[ERROR] Mongo delete failed: {e}")

    # 5) 파일 삭제 (uploads/<external_key>/)
    files_deleted = False
    try:
        if ek:
            uploads_base = Path(STATIC_DIR) / "uploads"
            files_deleted = _safe_delete_dir(uploads_base / ek, uploads_base)
    except Exception as e:
        print(f"[ERROR] file delete failed: {e}")
        files_deleted = False

    return {
        "message": "비디오 및 모든 관련 데이터 삭제 완료",
        "deleted_video_id": video_id,
        "history_deleted_count": history_deleted,
        "video_deleted": True,
        "mongo_deleted": mongo_deleted,
        "files_deleted": files_deleted,
    }

# 히스토리 생성 (기록)
@router.post("/{video_id}", response_model=HistoryResponse)
def create_history_entry(video_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    video = db.query(Video).filter_by(id=video_id, user_id=user.id).first()
    if not video:
        raise HTTPException(status_code=404, detail="해당 영상이 없습니다.")

    db_history = History(
        user_id=user.id,
        video_id=video_id,
        viewed_at=datetime.utcnow()
    )
    db.add(db_history)
    db.commit()
    return {"message": "히스토리 저장 완료", "video_id": video_id}


@router.get("/{video_id}", response_model=HistoryListResponse)
async def read_history(
    video_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    # 1) 권한/소유 확인
    video = db.query(Video).filter_by(id=video_id, user_id=user.id).first()
    if not video:
        raise HTTPException(status_code=404, detail="해당 영상이 없습니다.")

    # 2) Mongo에서 먼저 조회
    payload: Dict[str, Any] = {}
    try:
        coll = get_results_coll()
        doc = await coll.find_one({"video_id": video_id}, {"_id": 0})
        if doc:
            payload = doc
    except Exception:
        pass

    # 3) Mongo에 없으면 파일 폴백 (기존 로직)
    if not payload:
        key = video.external_key or (Path(video.video_name).stem if video.video_name else None)
        if key:
            p = Path(STATIC_DIR) / "uploads" / key / "result.json"
            if p.exists():
                payload = json.loads(p.read_text(encoding="utf-8"))

    return {
        "history": [{
            "video_id": video.id,
            "video_title": video.video_name,
            "upload_date": video.upload_date,
            "payload": payload, 
        }]
    }

# SAVE 전용: Mongo에 upsert (업로드 때는 저장 안 함)
# - 프론트가 payload를 보내면 그걸 사용
# - 안 보내면 서버가 result.json 읽어서 사용

class SaveHistoryBody(BaseModel):
    video_title: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


@router.post("/{video_id}/save")
async def save_history_and_upsert_mongo(
    video_id: str,
    body: SaveHistoryBody,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    print(f"[DEBUG] save 시작 - video_id: {video_id}, user_id: {user.id}")

    # 0) 명시적 거절: 'undefined'/'null'
    if video_id is None or video_id.lower() in ("undefined", "null"):
        raise HTTPException(status_code=400, detail="유효하지 않은 video_id 입니다.")

    actual_video_id: int
    video_row: Optional[Video] = None

    if video_id.isdigit():
        # 숫자 ID
        actual_video_id = int(video_id)
        video_row = db.query(Video).filter_by(id=actual_video_id, user_id=user.id).first()
        if not video_row:
            raise HTTPException(status_code=404, detail="해당 영상이 없습니다.")
        print(f"[INFO] 기존 DB 비디오 사용: {actual_video_id}")
    else:
        # 문자열 external_key (파일명 기반)
        ek = video_id.strip()
        if not ek:
            raise HTTPException(status_code=400, detail="external_key가 비어있습니다.")
        video_title = body.video_title or ek

        # 항상 새 레코드 생성 (덮어쓰기 방지)
        unique_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        new_external_key = f"{ek}-{unique_suffix}"

        new_video = Video(
            user_id=user.id,
            video_name=video_title,
            upload_date=datetime.utcnow().date(),
            external_key=new_external_key
        )
        db.add(new_video)
        db.flush()
        actual_video_id = new_video.id
        video_row = new_video
        print(f"[INFO] 새 비디오 생성(중복 허용): {actual_video_id}, ek={new_external_key}")
    db.commit()

    # 2) 히스토리
    h = History(user_id=user.id, video_id=actual_video_id, viewed_at=datetime.utcnow())
    db.add(h)
    db.commit()
    print(f"[INFO] 히스토리 생성 완료: user_id={user.id}, video_id={actual_video_id}")

    # 3) payload 준비
    doc: Dict[str, Any] = (body.payload or {}).copy()
    if not doc:
        # 파일 시스템 키는 항상 DB의 external_key로
        fs_key = (video_row.external_key if video_row and video_row.external_key else None)
        if not fs_key:
            print("[ERROR] external_key가 없어 result.json을 찾을 수 없음")
            raise HTTPException(status_code=400, detail="external_key를 찾을 수 없습니다.")
        p = Path(STATIC_DIR) / "uploads" / fs_key / "result.json"
        print(f"[DEBUG] result.json 경로: {p}")
        if p.exists():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                print(f"[DEBUG] result.json 로드 성공, 키 개수: {len(doc)}")
            except Exception as e:
                print(f"[ERROR] result.json 파싱 실패: {e}")
                doc = {}
        else:
            print(f"[WARN] result.json 파일이 존재하지 않음")

    if not doc:
        print(f"[ERROR] payload가 비어있음")
        raise HTTPException(status_code=400, detail="저장할 payload가 없습니다.")


    # Mongo에 upsert
    doc["video_id"] = int(actual_video_id)
    if body.video_title:
        doc["video_title"] = body.video_title

    for k in ("_id", "created_at", "updated_at"):
        doc.pop(k, None)

    try:
        coll = get_results_coll()
        result = await coll.update_one(
            {"video_id": int(actual_video_id)},
            {
                "$set": doc,
                "$setOnInsert": {"created_at": datetime.utcnow()},
                "$currentDate": {"updated_at": True},
            },
            upsert=True,
        )
        verification = await coll.find_one({"video_id": int(actual_video_id)})
        if not verification:
            print(f"[ERROR] 저장 검증 실패: 문서를 찾을 수 없음")
    except Exception as e:
        print(f"[ERROR] MongoDB 작업 중 오류: {e}")
        import traceback; print(traceback.format_exc())
        raise

    return {"ok": True, "video_id": actual_video_id, "saved_to_db": True}
