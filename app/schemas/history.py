from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Union

class Timeline(BaseModel):
    start_time: float
    end_time: float
    guideline_type: Union[str, int]

class HistoryResponse(BaseModel):
    video_id: int
    video_title: Optional[str] = None
    upload_date: Optional[datetime] = None
    timelines: Optional[List[Timeline]] = None

    class Config:
        orm_mode = True

class HistoryListResponse(BaseModel):
    history: List[HistoryResponse]

class HistoryDeleteResponse(BaseModel):
    message: str
    deleted_video_id: str