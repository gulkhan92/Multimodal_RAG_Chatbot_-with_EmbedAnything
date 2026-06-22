from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from rag_multimodal.settings import Settings
from rag_multimodal.database import SessionLocal, User as DBUser
import bcrypt

# Load environment variables
settings = Settings.from_env()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

class User(BaseModel):
    username: str
    email: Optional[str] = None
    roles: List[str] = ["viewer"]

    class Config:
        from_attributes = True # Pydantic V2 uses from_attributes instead of orm_mode

class TokenData(BaseModel):
    username: Optional[str] = None

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    db_user = db.query(DBUser).filter(DBUser.username == token_data.username).first()

    if db_user is None:
        raise credentials_exception
    
    return User(username=db_user.username, email=db_user.email, roles=[role.name for role in db_user.roles])

def RoleChecker(allowed_roles: List[str]):
    def _check(user: User = Depends(get_current_user)):
        if not any(role in user.roles for role in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource"
            )
        return user
    return _check

def get_password_hash(password):
    """Hashes a password using bcrypt, truncating it to 72 bytes for compatibility."""
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    
    hashed_bytes = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed_bytes.decode('utf-8')

def verify_password(plain_password, hashed_password):
    """Verifies a password against a bcrypt hash, truncating it to 72 bytes for compatibility."""
    password_bytes = plain_password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    
    hashed_password_bytes = hashed_password.encode('utf-8')
    
    return bcrypt.checkpw(password_bytes, hashed_password_bytes)