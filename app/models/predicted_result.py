from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.db.database import Base

class PredictedResult(Base):
    __tablename__ = "predicted_results"

    id = Column(Integer, primary_key=True, index=True)
    video_name = Column(String(255), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    labels = Column(String(2048))  # JSON 문자열로 저장
