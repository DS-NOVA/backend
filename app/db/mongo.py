from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

_MONGO_CLIENT: Optional[AsyncIOMotorClient] = None
_DB = None

def init_mongo():
    global _MONGO_CLIENT, _DB
    if _MONGO_CLIENT:
        return
    
    #env 파일 참고
    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB")

    _MONGO_CLIENT = AsyncIOMotorClient(uri)
    _DB = _MONGO_CLIENT[db_name]

def get_mongo_db():
    # 혹시 startup 전에 호출되면 지연 초기화
    if _DB is None:
        init_mongo()
    return _DB

def get_results_coll():
    # 컬렉션명 고정
    return get_mongo_db()["video_results"]