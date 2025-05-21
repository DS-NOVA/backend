from sqlalchemy import Column, VARCHAR, Integer
from app.db.database import Base

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(VARCHAR(100), unique=True, nullable=False)
    user_password= Column(VARCHAR(100), nullable=False)