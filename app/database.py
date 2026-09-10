import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

DB_PATH = os.environ.get("DB_PATH", "face_recognition.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database and creates tables if they do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT,
                face_descriptor TEXT NOT NULL,
                snapshot TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Audit logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                distance REAL NOT NULL,
                ip_address TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        conn.commit()


def create_user(
    username: str,
    full_name: str,
    email: Optional[str],
    face_descriptor: List[float],
    snapshot: Optional[str] = None
) -> Dict[str, Any]:
    """Registers a new user with their 128-d face descriptor vector."""
    descriptor_json = json.dumps(face_descriptor)
    created_at = datetime.now(timezone.utc).isoformat()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (username, full_name, email, face_descriptor, snapshot, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username.strip().lower(), full_name.strip(), email.strip() if email else "", descriptor_json, snapshot, created_at))
        conn.commit()
        user_id = cursor.lastrowid
        
    return {
        "id": user_id,
        "username": username.strip().lower(),
        "full_name": full_name.strip(),
        "email": email.strip() if email else "",
        "created_at": created_at
    }


def get_all_users(include_descriptor: bool = False) -> List[Dict[str, Any]]:
    """Retrieves all registered users."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name, email, face_descriptor, snapshot, created_at FROM users ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
    users = []
    for r in rows:
        user = {
            "id": r["id"],
            "username": r["username"],
            "full_name": r["full_name"],
            "email": r["email"],
            "snapshot": r["snapshot"],
            "created_at": r["created_at"]
        }
        if include_descriptor:
            user["face_descriptor"] = json.loads(r["face_descriptor"])
        users.append(user)
    return users


def get_user_by_username(username: str, include_descriptor: bool = False) -> Optional[Dict[str, Any]]:
    """Retrieves a single user by username."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name, email, face_descriptor, snapshot, created_at FROM users WHERE username = ?", (username.strip().lower(),))
        r = cursor.fetchone()
        
    if not r:
        return None
        
    user = {
        "id": r["id"],
        "username": r["username"],
        "full_name": r["full_name"],
        "email": r["email"],
        "snapshot": r["snapshot"],
        "created_at": r["created_at"]
    }
    if include_descriptor:
        user["face_descriptor"] = json.loads(r["face_descriptor"])
    return user


def delete_user(username: str) -> bool:
    """Deletes a user by username."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username.strip().lower(),))
        conn.commit()
        return cursor.rowcount > 0


def log_attempt(
    username: str,
    status: str,
    confidence: float,
    distance: float,
    ip_address: Optional[str] = None
):
    """Records an authentication attempt in audit logs."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (username, status, confidence, distance, ip_address, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, status, round(confidence, 2), round(distance, 4), ip_address or "unknown", timestamp))
        conn.commit()


def get_recent_logs(limit: int = 25) -> List[Dict[str, Any]]:
    """Fetches recent audit log entries."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, status, confidence, distance, ip_address, timestamp FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
    return [dict(r) for r in rows]


def get_system_stats() -> Dict[str, Any]:
    """Returns overview statistics for dashboard."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM users")
        total_users = cursor.fetchone()["count"]
        
        cursor.execute("SELECT COUNT(*) as count FROM audit_logs WHERE status = 'SUCCESS'")
        successful_logins = cursor.fetchone()["count"]
        
        cursor.execute("SELECT COUNT(*) as count FROM audit_logs WHERE status = 'FAILED'")
        failed_logins = cursor.fetchone()["count"]
        
    return {
        "total_users": total_users,
        "successful_logins": successful_logins,
        "failed_logins": failed_logins
    }
