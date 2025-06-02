import os
from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil

router = APIRouter(prefix="/nova/dashboard/video/upload") 

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_video(video: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, video.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        return {"message": f"File '{video.filename}' uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 업로드 실패: {str(e)}")

