# 👤 AURA.ID — Biometric Face Recognition Login System

A modern, high-performance biometric authentication web service and user login system powered by **Python FastAPI** and client-side neural vision inference.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

---

## 🌟 Key Features

- 🎯 **Real-time Face Detection & Biometric Extraction**: 68 facial landmark tracking and 128-dimensional embedding vector extraction directly in the browser via WebGL/WASM.
- ⚡ **Ultra-Low Memory (< 50 MB RAM)**: Specifically optimized to run flawlessly within **Render's Free Tier (512 MB RAM)** without out-of-memory crashes.
- 🔒 **Biometric Vector Matching**: Server-side Euclidean distance & confidence calculation with configurable thresholds.
- 🎨 **Futuristic Cyber HUD Interface**: Glowing reticles, laser scanning animations, and live video feedback with glassmorphism design.
- 📊 **Security Audit Trail**: Real-time logging of authentication attempts, confidence scores, distance metrics, and timestamps.
- 🧪 **Automated Test Suite**: Full test coverage with Pytest for all API endpoints.

---

## 🏗️ Architecture Overview

```
[ Browser Client ]
       │
       ├─► Webcam Video Feed
       ├─► Client-side Neural Inference (68 landmarks, 128-d vector)
       └─► POST /api/login or /api/register (Vector + Snapshot)
              │
              ▼
[ FastAPI Backend (Python 3.11 / Docker) ]
       │
       ├─► Vector Matcher (Euclidean Distance <= 0.50 Threshold)
       ├─► SQLite Database (Users, Descriptors, Audit Trail)
       └─► Session Token & Verification Response
```

---

## 🚀 One-Click Deploy to Render

1. Connect your repository [`laganarora/face-recognition-system`](https://github.com/laganarora/face-recognition-system) on **[Render](https://dashboard.render.com)**.
2. Select **Web Service**.
3. Choose the following settings:
   - **Language / Environment**: `Docker`
   - **Branch**: `main`
   - **Region**: `Singapore (Southeast Asia)`
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: **Free ($0 / month)**
4. Click **Deploy Web Service**!

---

## 💻 Local Development

### 1. Clone & Install
```bash
git clone https://github.com/laganarora/face-recognition-system.git
cd face-recognition-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Application
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at `http://127.0.0.1:8000` to access the Biometric Scanner.

### 3. Run Automated Tests
```bash
python -m pytest tests/test_api.py -v
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System healthcheck and user statistics |
| `POST` | `/api/register` | Enroll user with 128-d face descriptor vector |
| `POST` | `/api/login` | Authenticate face vector with 1-to-N matching |
| `GET` | `/api/users` | List enrolled identities |
| `DELETE` | `/api/users/{username}` | Delete user identity |
| `GET` | `/api/logs` | Security audit trail |
| `GET` | `/api/stats` | Dashboard statistics |

---

## 📄 License
MIT License
