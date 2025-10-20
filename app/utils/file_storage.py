import os
from uuid import uuid4
from fastapi import UploadFile, HTTPException
import aiofiles 

#회원가입 시 프로필 이미지 저장

ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
MAX_SIZE = 5 * 1024 * 1024  # 5MB

async def save_profile_image(upload: UploadFile, base_dir: str = "static/profiles"):
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="허용되지 않은 이미지 형식입니다.")

    os.makedirs(base_dir, exist_ok=True)

    filename = f"{uuid4().hex}{ALLOWED_TYPES[upload.content_type]}"
    disk_path = os.path.join(base_dir, filename)

    size = 0
    try:
        async with aiofiles.open(disk_path, "wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)  # 1MB씩
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_SIZE:
                    raise HTTPException(status_code=413, detail="파일이 너무 큽니다(최대 5MB).")
                await out.write(chunk)
    finally:
        await upload.close()

    # 브라우저에서 접근할 URL 경로로 변환
    return "/" + disk_path.replace(os.sep, "/")