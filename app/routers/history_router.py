from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import SessionLocal, get_db
from app.schemas.history import HistoryListResponse, HistoryDeleteResponse
from app.cruds.history_crud import get_history_by_video, delete_history_by_video

router = APIRouter(
    prefix="/nova/history",
    tags=["History"]
)

# 조회 API
@router.get("/{video_id}", response_model=HistoryListResponse)
def read_history(video_id: int, db: Session = Depends(get_db)):
    history = get_history_by_video(db, video_id)
    if not history:
        raise HTTPException(status_code=404, detail="히스토리가 없습니다.")
    return {"history": history}

# 삭제 API
@router.delete("/{video_id}", response_model=HistoryDeleteResponse)
def remove_history(video_id: int, db: Session = Depends(get_db)):
    deleted_count = delete_history_by_video(db, video_id)
    if not deleted_count:
        raise HTTPException(status_code=404, detail="삭제할 히스토리가 없습니다.")
    return {
        "message": "해당 영상 이력이 성공적으로 삭제되었습니다.",
        "deleted_video_id": str(video_id)
    }