from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserResponse
from app.cruds.user_crud import create_user, get_user_by_email, get_all_users
from app.db.database import SessionLocal
from app.security import verify_password
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


@router.post("/login", response_model=UserResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db:Session = Depends(get_db)):
    #유저가 있는지 확인해야 함
    user = get_user_by_email(db, form_data.username)#form data는 username으로 사용하므로 email 있어도 상관없음
    if not user:
        raise HTTPException(status_code=400, detail="존재하지 않는 계정입니다.")
    #잘못된 비밀번호 입력시
    if not verify_password(form_data.password, user.user_password):
        raise HTTPException(status_code=401, detail="비밀번호가 잘못 입력되었습니다.")
    return user

#모든 유저 조회
@router.get("/", response_model=list[UserResponse])
def read_all_users(db:Session = Depends(get_db)):
    return get_all_users(db)