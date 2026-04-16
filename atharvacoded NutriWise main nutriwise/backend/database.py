import sqlite3
import hashlib
from pathlib import Path

# Setup database path
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "backend" / "data" / "nutriwise.db"

# Create data directory if it doesn't exist
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def authenticate_or_register(email: str, password: str) -> dict:
    """
    Attempts to authenticate the user. 
    If the user does not exist, registers them automatically.
    Returns: {"status": "success" | "error", "message": "...", "user_id": int | None}
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    
    if row:
        user_id, stored_hash = row
        if stored_hash == _hash_password(password):
            conn.close()
            return {"status": "success", "message": "Login successful", "user_id": user_id}
        else:
            conn.close()
            return {"status": "error", "message": "Incorrect password", "user_id": None}
    else:
        # Auto-register
        try:
            hashed_pwd = _hash_password(password)
            cursor.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, hashed_pwd))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return {"status": "success", "message": "Account created and logged in", "user_id": user_id}
        except Exception as e:
            conn.close()
            return {"status": "error", "message": f"Database error: {str(e)}", "user_id": None}

# Initialize the db on import
init_db()
