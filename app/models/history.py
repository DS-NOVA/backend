from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database.mysql import Base
from datetime import datetime

class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    video_id = Column(Integer, ForeignKey("video.id"), nullable=False)
    viewed_at = Column(DateTime, default=datetime.utcnow)