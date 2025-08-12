from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserCreate
from app.security.security import get_password_hash

DEFAULT_USER_IMAGE = "/static/profiles/default.png"

#유저 생성 (회원가입)
#schema의 내용을 model에 input
def create_user(db:Session, new_user:UserCreate, image_url: str | None = None):
    user = User(
        user_name = new_user.user_name,
        user_image = image_url or DEFAULT_USER_IMAGE, #기본 프로필 이미지 (추후 수정)
        user_email = new_user.user_email,
        user_password = get_password_hash(new_user.user_password) #비밀번호 해싱
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

#유저 조회 (email)
def get_user_by_email(db:Session, email:str):
    return db.query(User).filter(User.user_email == email).first()

#유저 조회 (id)
def get_user_by_id(db: Session, user_id: int):
    return db.query(User).get(user_id)

#모든 유저 조회
def get_all_users(db:Session):
    return db.query(User).all()