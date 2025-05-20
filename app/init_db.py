# 데이터베이스 테이블 생성 시 필요

from app.db.database import Base, engine
#from app.models.test import Test #이 부분에 import 
from app.models.user import User

def init():
    print("nova 데이터베이스 생성중 ...")
    Base.metadata.create_all(bind=engine) #새로 import 된 테이블만 생성
    print("nova 데이터베이스 생성 완료")

if __name__ == "__main__":
    init()