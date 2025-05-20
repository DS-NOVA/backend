from sqlalchemy.orm import Session

from app.models.user import User
from app.schemas.user import UserCreate

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#유저 생성 (회원가입)
#schema의 내용을 model에 input
def create_user(db:Session, new_user:UserCreate):
    user = User(
        user_email = new_user.user_email,
        user_password = pwd_context.hash(new_user.user_password) #비밀번호 해싱
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

#유저 조희
def get_user_by_email(db:Session, email:str):
    return db.query(User).filter(User.user_email == email).first()

#모든 유저 조회
def get_all_users(db:Session):
    return db.query(User).all()