from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base

class VideoDetails(Base):
    __tablename__ = "video_details"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("video.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="video_details")