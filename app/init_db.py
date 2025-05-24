# 데이터베이스 테이블 생성 시 필요
from app.db.database import Base, engine
from app.models.history import History

def init():
    print("nova 데이터베이스 생성중 ...")
    Base.metadata.create_all(bind=engine)  # History 테이블 생성됨
    print("nova 데이터베이스 생성 완료")

if __name__ == "__main__":
    init()