from passlib.context import CryptContext
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from app.db.database import get_db
from app.models.user import User
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta 

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/nova/login")

SECRET_KEY = "test-secret-key"  
ALGORITHM = "HS256"              

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="유효하지 않은 사용자입니다.")
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="토큰이 유효하지 않습니다.")

# 테스트용 토큰
def create_test_token(user_id: int = 2, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + expires_delta
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

if __name__ == "__main__":
    token = create_test_token()
    print("테스트용 토큰:\n", token)