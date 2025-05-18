# 데이터베이스 확인을 위한 테스트 파일 (추후 삭제)
from sqlalchemy import Column, Integer, String
from app.db.database import Base

class Test(Base):
    __tablename__ = "test_table"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)