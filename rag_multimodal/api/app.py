from __future__ import annotations

import asyncio
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, status, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import func
from rag_multimodal.auth_utils import RoleChecker, create_access_token, verify_password, get_db, get_password_hash
from rag_multimodal.database import User as DBUser
from rag_multimodal.api.router import router as base_router
from rag_multimodal.api.router import build_chat_routes

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# ---- Routes ----

# Build chat routes (typed schemas + pydantic validation)
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


async def login(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    # Perform a case-insensitive username lookup to prevent login failures
    # due to capitalization (e.g., 'Admin' vs 'admin').
    user = db.query(DBUser).filter(func.lower(DBUser.username) == func.lower(form_data.username)).first()
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


async def signup(user: UserCreate, db=Depends(get_db)):
    from rag_multimodal.database import Role
    db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    viewer_role = db.query(Role).filter(Role.name == "viewer").first()
    if not viewer_role:
        raise HTTPException(status_code=500, detail="Default 'viewer' role not found")

    hashed_password = get_password_hash(user.password)
    new_user = DBUser(username=user.username, email=user.email, hashed_password=hashed_password, roles=[viewer_role])
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}


from rag_multimodal.auth_utils import RoleChecker


async def sync_data(_=Depends(RoleChecker(["admin"]))):
    # Placeholder for triggering sync_vector_db logic (keep behavior identical)
    return {"message": "Sync triggered successfully"}


async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(60)  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
