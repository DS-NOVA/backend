from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserResponse
from app.cruds.user_crud import create_user, get_user_by_email, get_all_users
from app.db.database import SessionLocal
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm

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

@router.get("/", response_model=list[UserResponse])
def read_all_users(db:Session = Depends(get_db)):
    return get_all_users(db)