import os
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserResponse, LoginResponse
from app.cruds.user_crud import create_user, get_user_by_email, get_all_users
from app.db.database import SessionLocal
from app.security import verify_password
from fastapi.security import OAuth2PasswordRequestForm
from dotenv import load_dotenv
from datetime import datetime, timedelta
from jose import jwt

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = float(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

router = APIRouter(prefix="/nova/auth")

def get_db():
    db = SessionLocal()
    try: 
        yield db 
    finally: 
        db.close()

@router.post("/signup", response_model=UserResponse)
def signup(new_user:UserCreate, db:Session = Depends(get_db)):
    #기존에 존재하는 회원인지 확인
    db_user = get_user_by_email(db, new_user.user_email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registerd")
    #없는 경우 회원가입 진행
    return create_user(db, new_user)


@router.post("/login", response_model=LoginResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db:Session = Depends(get_db), response: Response = None):
    #유저가 있는지 확인해야 함
    user = get_user_by_email(db, form_data.username)#form data는 username으로 사용하므로 email 있어도 상관없음

    if not user:
        raise HTTPException(status_code=400, detail="존재하지 않는 계정입니다.")
    #잘못된 비밀번호 입력시
    if not verify_password(form_data.password, user.user_password):
        raise HTTPException(status_code=401, detail="비밀번호가 잘못 입력되었습니다.")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.user_email}, expires_delta = access_token_expires)

    # 쿠키에 저장
    # secure 은 배포 시에는 반드시 True (https 환경에서만 전송)
    # samesite 는 cors 환경에 따라 조절
    # 프론트에서 axios.post('/login', data, { withCredentials: true }) 설정해야 함
    # 프론트에서 연동 후 response 에서 token 제거
    response.set_cookie(key="access_token", value=access_token, expires=access_token_expires, httponly=True, secure=False, samesite="lax")

    return {
        "message": "로그인 성공",
        "status_code": 200,
        "token": access_token,
        "user": user
    }

#모든 유저 조회
@router.get("/", response_model=list[UserResponse])
def read_all_users(db:Session = Depends(get_db)):
    return get_all_users(db)