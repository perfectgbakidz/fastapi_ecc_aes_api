# fastapi_ecc_aes_api_server_side_encrypt.py
"""
Full FastAPI ECC-AES Hybrid Medical Data Collector API
- Server-side ECC→AES encryption moved into /records/create
- Uses Argon2id for password hashing
- Includes: JWT auth (/token), user management, create/retrieve/view/update/delete records,
  key rotation, re-auth (password re-check) for dangerous actions, audit logging.

Run locally:
    pip install fastapi uvicorn cryptography pyjwt passlib[argon2] argon2-cffi
    uvicorn fastapi_ecc_aes_api_server_side_encrypt:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import base64
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List
import json
import secrets

# Crypto imports
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
import hmac

# Auth imports
import jwt
from passlib.context import CryptContext

# -------------------------------
# Configuration
# -------------------------------
DB_PATH = "medical_data.db"
KEYS_DIR = "keys"
SERVER_KEY_PATH = os.path.join(KEYS_DIR, "server_privkey.pem")
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

KDF_INFO = b"ecdh-aes256-gcm-medical"
AES_KEY_LEN = 32  # AES-256
NONCE_SIZE = 12

os.makedirs(KEYS_DIR, exist_ok=True)

# -------------------------------
# UPDATED: Argon2id password hashing
# -------------------------------

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__type="ID",            # Use Argon2id (recommended)
    argon2__memory_cost=65536,    # 64 MB RAM
    argon2__time_cost=3,
    argon2__parallelism=2,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
app = FastAPI(title="ECC-AES Hybrid Medical Data Collector API (Server-side Encryption) - Full")

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Database init
# -------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            client_pubkey TEXT,
            nonce_b64 TEXT,
            ciphertext_b64 TEXT NOT NULL,
            aad_b64 TEXT,
            note TEXT
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            username TEXT,
            action TEXT NOT NULL,
            target_id TEXT,
            details TEXT
        )
    ''')

    conn.commit()
    conn.close()

init_db()

# -------------------------------
# Key management
# -------------------------------

def generate_server_key(path: str) -> ec.EllipticCurvePrivateKey:
    priv = ec.generate_private_key(ec.SECP256R1(), default_backend())
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    with open(path, "wb") as f:
        f.write(priv_pem)
    return priv


def load_privkey(path: str) -> ec.EllipticCurvePrivateKey:
    with open(path, "rb") as f:
        pem = f.read()
    return serialization.load_pem_private_key(pem, password=None, backend=default_backend())

if not os.path.exists(SERVER_KEY_PATH):
    generate_server_key(SERVER_KEY_PATH)

SERVER_PRIV = load_privkey(SERVER_KEY_PATH)
SERVER_PUB = SERVER_PRIV.public_key()

# -------------------------------
# Utility crypto helpers
# -------------------------------

def pubkey_to_pem_b64(pubkey: ec.EllipticCurvePublicKey) -> str:
    pem = pubkey.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return base64.b64encode(pem).decode()


def load_pubkey_from_pem_b64(pem_b64: str) -> ec.EllipticCurvePublicKey:
    try:
        pem = base64.b64decode(pem_b64)
        pub = serialization.load_pem_public_key(pem, backend=default_backend())
        return pub
    except Exception as e:
        raise ValueError("Invalid client public key PEM/base64") from e


def derive_aes_key(shared_secret: bytes) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_LEN,
        salt=None,
        info=KDF_INFO,
        backend=default_backend()
    )
    return hkdf.derive(shared_secret)

def constant_time_compare(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)

# -------------------------------
# Auth & user management
# -------------------------------

def get_user(username: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT username, hashed_password, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"username": row[0], "hashed_password": row[1], "role": row[2]}


def create_user(username: str, password: str, role: str = "clinician"):
    hashed = pwd_context.hash(password)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO users (username, hashed_password, role) VALUES (?, ?, ?)", (username, hashed, role))
    conn.commit()
    conn.close()


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM users")
count = cur.fetchone()[0]
conn.close()
if count == 0:
    print("No users found - creating default admin user: 'admin' with password 'adminpass' (change immediately)")
    create_user("admin", "adminpass", role="admin")


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user:
        return None
    if not pwd_context.verify(password, user["hashed_password"]):
        return None
    return user


def verify_user_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    user = get_user(username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

async def require_role(role: str, current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != role and current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient privileges")
    return current_user

# -------------------------------
# Pydantic models
# -------------------------------

class ReAuthRequest(BaseModel):
    password: str

# UPDATED: accept plaintext. Server will encrypt.
class RecordCreateRequest(BaseModel):
    plaintext: str
    note: Optional[str] = None

class RecordUpdateRequest(BaseModel):
    note: Optional[str] = None

class RotateKeyRequest(BaseModel):
    password: str

# -------------------------------
# Audit logging
# -------------------------------

def log_audit(username: Optional[str], action: str, target_id: Optional[str] = None, details: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO audit_logs (timestamp, username, action, target_id, details) VALUES (?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), username, action, target_id, details))
    conn.commit()
    conn.close()

# -------------------------------
# Re-auth helper
# -------------------------------

def require_reauth(current_user: dict, reauth_password: str):
    if not verify_user_password(reauth_password, current_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Re-authentication failed")

# -------------------------------
# Token endpoint
# -------------------------------

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token({"sub": user["username"], "role": user["role"]}, expires_delta=access_token_expires)
    log_audit(user["username"], "login", None, "issued JWT token")
    return {"access_token": token, "token_type": "bearer"}

# -------------------------------
# User management endpoints
# -------------------------------

@app.post("/users/create")
async def api_create_user(username: str, password: str, role: str = "clinician", reauth: ReAuthRequest = Depends(), current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admin can create users")
    require_reauth(current_user, reauth.password)
    create_user(username, password, role)
    log_audit(current_user["username"], "create_user", username, f"role={role}")
    return {"status": "user_created", "username": username}

# -------------------------------
# Record endpoints (server-side encryption)
# -------------------------------

@app.post("/records/create")
async def create_record(req: RecordCreateRequest, current_user: dict = Depends(get_current_user)):
    if current_user['role'] not in ["clinician", "doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Insufficient privileges to create records")

    # --- SERVER-SIDE ENCRYPTION FLOW ---
    # Generate an ephemeral EC keypair (we will store the public key in client_pubkey column)
    ephemeral_priv = ec.generate_private_key(ec.SECP256R1(), default_backend())
    ephemeral_pub = ephemeral_priv.public_key()

    # Compute shared secret between SERVER_PRIV (static) and ephemeral_pub
    # Note: This is symmetric with ephemeral_priv.exchange(SERVER_PUB), but we're using SERVER_PRIV.exchange(ephemeral_pub)
    shared = SERVER_PRIV.exchange(ec.ECDH(), ephemeral_pub)

    # Derive AES-256 key via HKDF
    aes_key = derive_aes_key(shared)
    aesgcm = AESGCM(aes_key)

    # Encrypt plaintext with AES-GCM
    nonce = os.urandom(NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, req.plaintext.encode(), None)

    # Prepare stored fields (base64 encoded)
    client_pubkey_b64 = pubkey_to_pem_b64(ephemeral_pub)
    nonce_b64 = base64.b64encode(nonce).decode()
    ciphertext_b64 = base64.b64encode(ciphertext).decode()
    aad_b64 = None

    # Store encrypted record in DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO records (created_at, client_pubkey, nonce_b64, ciphertext_b64, aad_b64, note) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), client_pubkey_b64, nonce_b64, ciphertext_b64, aad_b64, req.note))
    record_id = cur.lastrowid
    conn.commit()
    conn.close()

    log_audit(current_user['username'], "create_record", str(record_id), f"note={req.note}")

    # Return success (we do NOT return plaintext or AES key)
    return {"status": "created", "record_id": record_id}

@app.post("/records/{record_id}")
async def view_record(record_id: int, reauth: ReAuthRequest, current_user: dict = Depends(get_current_user)):
    if current_user['role'] not in ["doctor", "clinician", "admin"]:
        raise HTTPException(status_code=403, detail="Access denied")

    require_reauth(current_user, reauth.password)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, created_at, client_pubkey, nonce_b64, ciphertext_b64, aad_b64, note FROM records WHERE id = ?", (record_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Record not found")

    try:
        client_pub = load_pubkey_from_pem_b64(row[2])
        shared = SERVER_PRIV.exchange(ec.ECDH(), client_pub)
        aes_key = derive_aes_key(shared)
        aesgcm = AESGCM(aes_key)
        # row[3] is nonce_b64, row[4] is ciphertext_b64
        plaintext = aesgcm.decrypt(base64.b64decode(row[3]), base64.b64decode(row[4]), None)
    except Exception as e:
        log_audit(current_user['username'], "view_record_failed", str(record_id), str(e))
        raise HTTPException(status_code=500, detail=f"Decryption failed: {e}")

    log_audit(current_user['username'], "view_record", str(record_id), "success")

    return {
        "id": row[0],
        "created_at": row[1],
        "client_pubkey": row[2],
        "note": row[6],
        "plaintext": plaintext.decode()
    }


@app.post("/records/{record_id}/retrieve")
async def retrieve_record(record_id: int, reauth: ReAuthRequest, current_user: dict = Depends(get_current_user)):
    return await view_record(record_id, reauth, current_user)


@app.put("/records/{record_id}")
async def update_record(record_id: int, payload: RecordUpdateRequest, reauth: ReAuthRequest,
                        current_user: dict = Depends(get_current_user)):
    require_reauth(current_user, reauth.password)

    if current_user["role"] not in ["clinician", "doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Insufficient privileges")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM records WHERE id = ?", (record_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Record not found")

    cur.execute("UPDATE records SET note = ? WHERE id = ?", (payload.note, record_id))
    conn.commit()
    conn.close()

    log_audit(current_user['username'], "update_record", str(record_id), f"note={payload.note}")

    return {"status": "updated", "record_id": record_id}


@app.delete("/records/{record_id}")
async def delete_record(record_id: int, reauth: ReAuthRequest, current_user: dict = Depends(get_current_user)):
    require_reauth(current_user, reauth.password)

    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admin can delete records")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM records WHERE id = ?", (record_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Record not found")

    cur.execute("DELETE FROM records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

    log_audit(current_user['username'], "delete_record", str(record_id), "deleted")

    return {"status": "deleted", "record_id": record_id}

# -------------------------------
# Key rotation
# -------------------------------

@app.post("/keys/rotate")
async def rotate_keys(req: RotateKeyRequest, current_user: dict = Depends(get_current_user)):
    require_reauth(current_user, req.password)
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admin can rotate keys")

    generate_server_key(SERVER_KEY_PATH)

    global SERVER_PRIV, SERVER_PUB
    SERVER_PRIV = load_privkey(SERVER_KEY_PATH)
    SERVER_PUB = SERVER_PRIV.public_key()

    log_audit(current_user['username'], "rotate_keys", None, "server key rotated")

    return {"status": "success", "message": "Server ECC key rotated (existing records not re-encrypted)"}

# -------------------------------
# Utility endpoints
# -------------------------------

@app.get("/records")
async def list_records(current_user: dict = Depends(get_current_user)):
    if current_user['role'] not in ["clinician", "doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Insufficient privileges")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, created_at, note FROM records ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()

    return [{"id": r[0], "created_at": r[1], "note": r[2]} for r in rows]


@app.get("/audit_logs")
async def get_audit_logs(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admin can view audit logs")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, username, action, target_id, details FROM audit_logs ORDER BY timestamp DESC LIMIT 1000")
    rows = cur.fetchall()
    conn.close()

    return [{"id": r[0], "timestamp": r[1], "username": r[2], "action": r[3], "target_id": r[4], "details": r[5]} for r in rows]


# -------------------------------
# Server public key
# -------------------------------

@app.get("/keys/server_pub")
async def get_server_pub():
    return {"server_pub_b64": pubkey_to_pem_b64(SERVER_PUB)}
