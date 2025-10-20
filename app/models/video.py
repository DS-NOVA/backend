from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base

class Video(Base):
    __tablename__ = "video"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    video_name = Column(String(255), nullable=False)
    upload_date = Column(Date)
    external_key = Column(String(128), unique=True, index=True, nullable=True) #몽고디비와 사용할 키
    
    user = relationship("User", back_populates="videos")
    histories = relationship("History", back_populates="video")
    video_details = relationship("VideoDetails", back_populates="video")