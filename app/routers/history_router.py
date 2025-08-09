from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import SessionLocal, get_db
from app.schemas.history import HistoryResponse
from app.schemas.history import HistoryListResponse, HistoryDeleteResponse
from app.models.history import History as history
from app.security import get_current_user
from app.cruds.history_crud import get_history_by_video, delete_history_by_video
from app.models.video import Video
from datetime import datetime

router = APIRouter(
    prefix="/nova/history",
    tags=["History"]
)

# 조회 API
@router.get("/{video_id}", response_model=HistoryListResponse)
def read_history(video_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    video = db.query(Video).filter_by(id=video_id, user_id=user.id).first()
    if not video:
        raise HTTPException(status_code=404, detail="해당 영상이 없습니다.")
    flash_result = db.query(FlashResult).filter_by(video_id=video_id).first()
    timelines = db.query(Timeline).filter_by(video_id=video_id).all()
    return {
        "history": [{
            "video_id": video.id,
            "video_title": video.filename,
            "upload_date": video.created_at.strftime("%Y-%m-%d"),
            "flash_count": flash_result.flash_count if flash_result else 0,
            "timelines": [t.to_list() for t in timelines]  # or reshape as needed
        }]
    }

# 삭제 API
@router.delete("/{video_id}", response_model=HistoryDeleteResponse)
def remove_history(
    video_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    video = db.query(Video).filter_by(id=video_id, user_id=user.id).first()
    if not video:
        raise HTTPException(status_code=403, detail="삭제 권한이 없습니다.")

    deleted_count = delete_history_by_video(db, video_id)
    if not deleted_count:
        raise HTTPException(status_code=404, detail="삭제할 히스토리가 없습니다.")

    return {
        "message": "해당 영상 이력이 성공적으로 삭제되었습니다.",
        "deleted_video_id": video_id
    }

@router.post("/{video_id}", response_model=HistoryResponse)
def create_history_entry(video_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    video = db.query(Video).filter_by(id=video_id, user_id=user.id).first()
    if not video:
        raise HTTPException(status_code=404, detail="해당 영상이 없습니다.")

    db_history = history(
        user_id=user.id,
        video_id=video_id,
        viewed_at=datetime.utcnow()
    )
    db.add(db_history)
    db.commit()
    return {"message": "히스토리 저장 완료", "video_id": video_id}