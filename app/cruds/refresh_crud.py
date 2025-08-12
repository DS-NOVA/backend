from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from app.models.refresh_token import RefreshToken
from fastapi import HTTPException
import secrets, base64, hashlib, os
from dotenv import load_dotenv

load_dotenv()

#토큰 유효 기간
REFRESH_TTL_DAYS = 14
REFRESH_BYTES = 64
REFRESH_PEPPER = os.getenv("REFRESH_PEPPER") #토큰 생성할 키값


if not REFRESH_PEPPER:
    raise RuntimeError("REFRESH_PEPPER is not set. Add it to your .env for secure refresh tokens.")


def remove_padding(raw: bytes):
    #바이트 -> 문자열
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

def token_to_hash(token_str: str):
    # token_str은 base64url 문자열
    hash = hashlib.sha256()
    hash.update(token_str.encode("utf-8"))
    hash.update(REFRESH_PEPPER.encode("utf-8"))
    return hash.hexdigest()

def create_refresh_token(db: Session, user_id: int):
    raw = secrets.token_bytes(REFRESH_BYTES)
    token = remove_padding(raw) #문자열로 변환 -> 쿠키로 사용할 것
    token_hash = token_to_hash(token) #인코딩 -> db에 저장될 값

    #db에는 해시만 저장
    expires = datetime.now(timezone.utc) + timedelta(days=REFRESH_TTL_DAYS)
    rec = RefreshToken(user_id=user_id, token_hash=token_hash, expires_at=expires, revoked=False)
    db.add(rec)
    db.commit()
    return token

#토큰 조회하기 
#create_refresh_token에서 db에 저장한 token_hash와 비교하기
def get_refresh_token_from_db(db: Session, token_str: str):
    token_hash = token_to_hash(token_str)
    return db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()

#토큰 무효화하기
def revoke_refresh_token(db: Session, token_str: str):
    token_revoke = get_refresh_token_from_db(db, token_str)
    if token_revoke:
        token_revoke.revoked = True
        db.commit()

#토큰 삭제하기 (로그아웃)
def delete_refresh_token_for_user(db: Session, user_id: int):
    db.query(RefreshToken).filter(RefreshToken.user_id == user_id).delete()
    db.commit()

#토큰 검사하기
def verify_refresh_token(db: Session, token_str: str):
    token_get = get_refresh_token_from_db(db, token_str)
    #무효화된 경우
    if not token_get or token_get.revoked:
        raise HTTPException(status_code=401, detail={"error": "token_revoked"})
    #만료된 경우
    expires_at = token_get.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        token_get.revoked = True
        db.commit()
        raise HTTPException(status_code=401, detail={"error": "token_revoked"})
    return token_get.user_id

#토큰 재발급을 위해 이전 토큰 무효화하기
def rotate_refresh_token(db: Session, old_token_str: str, user_id: int):
    revoke_refresh_token(db, old_token_str)
    return create_refresh_token(db, user_id)