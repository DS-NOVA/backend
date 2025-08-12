import os
from fastapi import APIRouter, Depends, HTTPException, Response, Request, Header, Form, File, UploadFile
import secrets
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserResponse, LoginResponse
from app.cruds.user_crud import create_user, get_user_by_email, get_all_users
from app.db.database import SessionLocal, get_db
from app.security.security import verify_password
from app.security.auth import create_access_token, get_current_user
from fastapi.security import OAuth2PasswordRequestForm
from dotenv import load_dotenv
from datetime import timedelta
from app.cruds.refresh_crud import *
from app.utils.file_storage import save_profile_image



load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = float(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

router = APIRouter(prefix="/nova/auth")

#회원가입
@router.post("/signup", response_model=UserResponse)
async def signup(user_name: str = Form(...), user_email: str = Form(...), user_password: str = Form(...), user_image: UploadFile | None = File(None),
    db: Session = Depends(get_db)):
    #기존에 존재하는 회원인지 확인 (이메일로)
    db_user = get_user_by_email(db, user_email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registerd")
    
    image_url: str | None = None
    if user_image:
        image_url = await save_profile_image(user_image)

    new_user = UserCreate(
            user_name = user_name,
            user_email = user_email,
            user_password = user_password
        )
    
    return create_user(db, new_user, image_url=image_url)

#로그인
@router.post("/login", response_model=LoginResponse)
def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db:Session = Depends(get_db)):
    #유저가 있는지 확인해야 함
    user = get_user_by_email(db, form_data.username)#form data는 username으로 사용하므로 email 있어도 상관없음

    if not user:
        raise HTTPException(status_code=400, detail="존재하지 않는 계정입니다.")
    #잘못된 비밀번호 입력시
    if not verify_password(form_data.password, user.user_password):
        raise HTTPException(status_code=401, detail="비밀번호가 잘못 입력되었습니다.")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": str(user.id)}, expires_delta = access_token_expires)

    # 쿠키에 저장
    # secure 은 배포 시에는 반드시 True (https 환경에서만 전송)
    # samesite 는 cors 환경에 따라 조절
    # secure=True, samesite="none"
    refresh_token = create_refresh_token(db, user.id)
    csrf_token = secrets.token_urlsafe(16)

    response.set_cookie(
        "refresh_token", refresh_token, httponly=True, secure=False, samesite="lax", max_age=14*24*3600, path="/nova/auth")
    response.set_cookie(
        "csrf_token", csrf_token, httponly=False, secure=False, samesite="lax", max_age=14*24*3600, path="/")#csrf 토큰은 모든 경로에서 필요하기 때문에 루트 경로로

    return {
        "message": "로그인 성공",
        "status_code": 200,
        "access_token": access_token,
        "user": user
    }

#모든 유저 조회 (추후 삭제)
@router.get("/", response_model=list[UserResponse])
def read_all_users(db:Session = Depends(get_db)):
    return get_all_users(db)


#refresh 토큰, csrf 쿠키 읽기
@router.post("/refresh")
def refresh(request: Request, response: Response, x_csrf_token: str = Header(..., alias="X-CSRF-Token"), db: Session = Depends(get_db)):
    refresh_token = request.cookies.get("refresh_token")
    csrf_cookie = request.cookies.get("csrf_token")
    #csrf_header = request.headers.get("x-csrf-token")

    #리프레시 토큰이 없는 경우 (401) -> 로그인이 되어있지 않음 (로그아웃 시 토큰 삭제)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="no_refresh_token")

    #csrf 검증 실패한 경우 (403)
    if not csrf_cookie or csrf_cookie != x_csrf_token:
        raise HTTPException(status_code=403, detail="CSRF check failed")

    #토큰 꺼내오기
    rec = get_refresh_token_from_db(db, refresh_token)
    if not rec:
        # 서버 저장소에 없는 토큰 -> revoke 취급 (만료됨)
        raise HTTPException(status_code=401, detail={"error": "token_revoked"})

    user_id = verify_refresh_token(db, refresh_token)  # int user_id 반환(만료/취소 검사 포함)

    #refresh가 호출됨 = Access 재발급 필요
    new_access = create_access_token({"sub": str(user_id)},expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    # 새로운 access 발급 -> 기존 refresh 무효화로 탈취 대비
    new_refresh = rotate_refresh_token(db, refresh_token, user_id)
    response.set_cookie("refresh_token", new_refresh, httponly=True, secure=False, samesite="lax", max_age=14 * 24 * 3600, path="/nova/auth")

    return {"access_token": new_access, "token_type": "bearer"}

#로그아웃
@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    rt = request.cookies.get("refresh_token")

    if rt:
        try:
            user_id = verify_refresh_token(db, rt)  # 실패 시 예외
            delete_refresh_token_for_user(db, user_id)
        except HTTPException:
            pass

    # 쿠키 삭제 (항상)
    response.delete_cookie("refresh_token", path="/nova/auth")
    response.delete_cookie("csrf_token", path="/")
    return {"message": "logged out"}


#테스트용 라우터 (추후 삭제 혹은 경로 수정)
@router.get("/me")
def me(user = Depends(get_current_user)):
    # 필요한 최소 정보만 반환 (스키마 맞추고 싶으면 UserResponse 사용해도 됨)
    return {
        "ok": True,
        "user": {
            "id": user.id,
            "user_name": user.user_name,
            "user_email": user.user_email,
        }
    }

