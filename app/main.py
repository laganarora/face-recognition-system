import os
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.database import (
    init_db,
    create_user,
    get_all_users,
    get_user_by_username,
    delete_user,
    log_attempt,
    get_recent_logs,
    get_system_stats
)
from app.matcher import find_best_match, euclidean_distance, calculate_confidence, DEFAULT_THRESHOLD

# Initialize Database on app start
init_db()

app = FastAPI(
    title="Face Recognition Authentication API",
    description="High-performance biometric authentication web service powered by FastAPI",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Models ---
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    full_name: str = Field(..., min_length=1, max_length=100)
    email: Optional[str] = None
    face_descriptor: List[float]
    snapshot: Optional[str] = None

    @field_validator("face_descriptor")
    @classmethod
    def validate_descriptor(cls, v):
        if len(v) != 128:
            raise ValueError(f"Face descriptor must have exactly 128 dimensions, got {len(v)}")
        return v


class LoginRequest(BaseModel):
    face_descriptor: List[float]
    username: Optional[str] = None
    snapshot: Optional[str] = None

    @field_validator("face_descriptor")
    @classmethod
    def validate_descriptor(cls, v):
        if len(v) != 128:
            raise ValueError(f"Face descriptor must have exactly 128 dimensions, got {len(v)}")
        return v


# --- API Routes ---

@app.get("/api/health")
def health_check():
    """Healthcheck endpoint for Render zero-downtime deployment."""
    stats = get_system_stats()
    return {
        "status": "healthy",
        "service": "face-recognition-system",
        "version": "1.0.0",
        "registered_users": stats["total_users"]
    }


@app.get("/api/stats")
def stats_endpoint():
    """Retrieve application overview stats."""
    return get_system_stats()


@app.post("/api/register", status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, request: Request):
    """Registers a new user with face biometric descriptor."""
    username_clean = req.username.strip().lower()
    
    # Check if username already exists
    existing = get_user_by_username(username_clean)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{username_clean}' is already registered."
        )
        
    user = create_user(
        username=username_clean,
        full_name=req.full_name,
        email=req.email,
        face_descriptor=req.face_descriptor,
        snapshot=req.snapshot
    )
    
    client_ip = request.client.host if request.client else "unknown"
    log_attempt(
        username=username_clean,
        status="REGISTERED",
        confidence=100.0,
        distance=0.0,
        ip_address=client_ip
    )
    
    return {
        "success": True,
        "message": f"User '{username_clean}' registered successfully.",
        "user": user
    }


@app.post("/api/login")
def login(req: LoginRequest, request: Request):
    """
    Authenticates a user via face biometric matching.
    Supports both 1-to-N matching (automatic identification)
    and 1-to-1 verification (if username is specified).
    """
    client_ip = request.client.host if request.client else "unknown"
    
    # If a specific username is requested for 1-to-1 matching
    if req.username:
        username_clean = req.username.strip().lower()
        user = get_user_by_username(username_clean, include_descriptor=True)
        if not user:
            log_attempt(
                username=username_clean,
                status="FAILED",
                confidence=0.0,
                distance=1.0,
                ip_address=client_ip
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{username_clean}' not found."
            )
            
        distance = euclidean_distance(req.face_descriptor, user["face_descriptor"])
        confidence = calculate_confidence(distance, DEFAULT_THRESHOLD)
        
        if distance <= DEFAULT_THRESHOLD:
            log_attempt(
                username=username_clean,
                status="SUCCESS",
                confidence=confidence,
                distance=distance,
                ip_address=client_ip
            )
            session_token = f"sess_{uuid.uuid4().hex}"
            return {
                "success": True,
                "authenticated": True,
                "user": {
                    "username": user["username"],
                    "full_name": user["full_name"],
                    "email": user["email"],
                    "snapshot": user["snapshot"]
                },
                "confidence": round(confidence, 1),
                "distance": round(distance, 4),
                "session_token": session_token
            }
        else:
            log_attempt(
                username=username_clean,
                status="FAILED",
                confidence=confidence,
                distance=distance,
                ip_address=client_ip
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Face mismatch. Confidence ({confidence:.1f}%) is below required threshold."
            )

    # 1-to-N Matching: Compare against all registered users
    all_users = get_all_users(include_descriptor=True)
    if not all_users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No registered users found in the system. Please register a face first."
        )
        
    best_user, distance, confidence = find_best_match(
        query_descriptor=req.face_descriptor,
        users=all_users,
        threshold=DEFAULT_THRESHOLD
    )
    
    if best_user:
        log_attempt(
            username=best_user["username"],
            status="SUCCESS",
            confidence=confidence,
            distance=distance,
            ip_address=client_ip
        )
        session_token = f"sess_{uuid.uuid4().hex}"
        return {
            "success": True,
            "authenticated": True,
            "user": {
                "username": best_user["username"],
                "full_name": best_user["full_name"],
                "email": best_user["email"],
                "snapshot": best_user["snapshot"]
            },
            "confidence": round(confidence, 1),
            "distance": round(distance, 4),
            "session_token": session_token
        }
    else:
        log_attempt(
            username="Unknown",
            status="FAILED",
            confidence=confidence,
            distance=distance,
            ip_address=client_ip
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No matching face recognized in the database. Please align your face or register first."
        )


@app.get("/api/users")
def list_users():
    """Lists all registered users (descriptors hidden for security)."""
    return get_all_users(include_descriptor=False)


@app.delete("/api/users/{username}")
def remove_user(username: str):
    """Deletes a registered user."""
    deleted = delete_user(username)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found."
        )
    return {"success": True, "message": f"User '{username}' removed."}


@app.get("/api/logs")
def audit_logs(limit: int = 25):
    """Fetches recent authentication logs."""
    return get_recent_logs(limit=limit)


# Mount static directory to serve frontend web UI
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
