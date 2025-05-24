from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base

class Video(Base):
    __tablename__ = "video"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    video_name = Column(String(255), nullable=False)
    sensitivity_level = Column(Integer)
    upload_date = Column(Date)

    user = relationship("User", back_populates="videos")
    histories = relationship("History", back_populates="video")
    details = relationship("VideoDetails", back_populates="video")