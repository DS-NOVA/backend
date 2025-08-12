from sqlalchemy import Column, VARCHAR, Integer
from sqlalchemy.orm import relationship
from app.db.database import Base
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(VARCHAR(100), nullable=False)
    user_image = Column(VARCHAR(200), nullable=True)
    user_email = Column(VARCHAR(100), unique=True, nullable=False)
    user_password= Column(VARCHAR(100), nullable=False)

    videos = relationship("Video", back_populates="user")
    histories = relationship("History", back_populates="user")