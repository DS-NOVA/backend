from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.user import User
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
from jose.exceptions import ExpiredSignatureError

# 프론트에서 토큰 가져오기
#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/nova/auth/login")
bearer_scheme = HTTPBearer()

# 환경 변수에서 비밀 키와 알고리즘 가져오기
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

#토큰 생성하기 
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    now = datetime.now(timezone.utc)  # 현재 UTC (aware)
    expire = now + (expires_delta or timedelta(minutes=15))
    to_encode.update({
        "exp": expire,
        "iat": now,
        "nbf": now,
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

#JWT 토큰 디코딩 및 검증
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme), db: Session = Depends(get_db)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise_token_invalid()
        user_id = int(sub)
    except ExpiredSignatureError:
        raise_token_expired()
    except JWTError:
        raise_token_invalid()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise_token_invalid()
    return user


def raise_token_expired():
    # 표준 헤더 + 에러 코드 바디 모두 제공 (프론트가 어느 쪽이든 읽을 수 있게)
    headers = {'WWW-Authenticate': 'Bearer error="invalid_token", error_description="The access token expired"'}
    body = {"error": "token_expired", "detail": "access token expired"}
    raise HTTPException(status_code=401, detail=body, headers=headers)

def raise_token_invalid():
    headers = {'WWW-Authenticate': 'Bearer error="invalid_token", error_description="The access token is invalid"'}
    body = {"error": "token_invalid", "detail": "invalid or malformed access token"}
    raise HTTPException(status_code=401, detail=body, headers=headers)