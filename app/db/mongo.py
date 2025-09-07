from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
import os

_MONGO_CLIENT: Optional[AsyncIOMotorClient] = None
_DB = None

def init_mongo():
    global _MONGO_CLIENT, _DB
    if _MONGO_CLIENT:
        return
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "nova")
    _MONGO_CLIENT = AsyncIOMotorClient(uri)
    _DB = _MONGO_CLIENT[db_name]

def get_mongo_db():
    if _DB is None:
        # 혹시 startup 전에 호출되면 지연 초기화
        init_mongo()
    return _DB

def get_results_coll():
    return get_mongo_db()["video_results"]  # 컬렉션명 고정
