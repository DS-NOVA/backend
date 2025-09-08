from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any


class HistoryResponse(BaseModel):
    video_id: int
    video_title: Optional[str] = None
    upload_date: Optional[datetime] = None
    payload: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True

class HistoryListResponse(BaseModel):
    history: List[HistoryResponse]

class HistoryDeleteResponse(BaseModel):
    message: str
    deleted_video_id: int
    history_deleted_count: int
    video_deleted: bool
    mongo_deleted: int
    files_deleted: bool