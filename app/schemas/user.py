from pydantic import BaseModel, EmailStr, Field, field_validator

#user 생성 스키마 (회원가입 시 사용)
class UserCreate(BaseModel):
    id: int
    user_email: EmailStr
    user_password: str = Field(min_length=6)

"""
유효성 검사
->이메일, 패스워드가 하나라도 입력되지 않으면 not_empty 출력
오류 발생 시 422 에러 반환
"""

@field_validator('user_email, user_password')
def not_empty(cls, v, field):
    if not v or not v.strip():
        raise ValueError(f"{field.name} 을(를) 입력해주세요.")
    return v

#user response 스키마
class UserResponse(BaseModel):
    id: int
    user_email: EmailStr

    class Config:
        orm_model = True