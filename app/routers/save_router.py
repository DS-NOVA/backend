import os
from fastapi import APIRouter, HTTPException, Body
from pymongo import MongoClient
from datetime import datetime

'''
변환된 영상 json 데이터를 MongoDB에 저장
'''

router = APIRouter(prefix="/nova/dashboard/save")

# MongoDB 연결
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["conversion_db"]
collection = db["conversion_history"]

@router.post("/")
async def save_converted_video(result: dict = Body(...)):
    try:
        result["created_at"] = datetime.utcnow()
        insert_result = collection.insert_one(result)
        return {
            "message": f"{result.get('video_id', 'unknown')} 저장 완료!",
            "mongo_id": str(insert_result.inserted_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장 중 오류 발생: {str(e)}")
