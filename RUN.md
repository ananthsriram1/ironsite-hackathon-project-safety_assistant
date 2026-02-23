# IronSite — Step-by-Step Run Guide

This guide covers running the app **on your GPU instance** (recommended) and **locally** (frontend only, pointing to the GPU backend).

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for frontend) or **Bun** (optional)
- **GPU instance**: CUDA-capable (e.g. Vast.ai). Models are expected at `/workspace/models/` on the instance.

---

## Option A: Run everything on the GPU instance (recommended)

The backend needs YOLO, SAM3, and Qwen3-VL; these run best on a GPU. Use this flow to run both backend and frontend on the same machine.

### 1. SSH into your GPU instance

```bash
# Replace with your actual SSH (e.g. from Vast.ai or your provider)
ssh -L 8000:localhost:8000 -L 3000:localhost:3000 YOUR_USER@YOUR_GPU_HOST
```

Port forwarding (`-L`) lets you open the app in your browser on your Mac while the server runs on the GPU box.

### 2. Clone/copy the project on the GPU instance

If the repo isn’t there yet:

```bash
# Example: clone or rsync from your Mac
git clone <your-repo-url> /workspace/ironsite
# or from your Mac: rsync -avz --exclude venv --exclude node_modules ./ironsite-hackathon-project-safety_assistant/ user@host:/workspace/ironsite/
```

### 3. Model weights on the GPU instance

Ensure these exist **before** starting the backend:

| File / directory | Location (on GPU instance) |
|------------------|----------------------------|
| YOLO (fine-tuned) | `backend/models/last.pt` |
| SAM3 | `/workspace/models/sam3.pt` |
| Qwen3-VL base | `/workspace/models/qwen3-vl` (HuggingFace dir) |
| LoRA adapter (optional) | `/workspace/models/adapter1` |

Create the backend models dir if needed:

```bash
mkdir -p backend/models
# Then copy last.pt into backend/models/
```

### 4. Backend setup and run (on GPU instance)

**One-time setup:**

```bash
cd /workspace/ironsite/backend   # or your project path
bash setup.sh
```

On Linux you can optionally install `decord` for video tooling:

```bash
source venv/bin/activate && pip install decord
```

**Start the API server:**

```bash
cd backend
source venv/bin/activate
# Use the GPUs you want (e.g. 0,1,3). Omit or set to 0 for single GPU.
CUDA_VISIBLE_DEVICES=0,1,3 uvicorn main:app --host 0.0.0.0 --port 8000
```

- Backend: **http://localhost:8000** (or the GPU host’s IP if you’re not using SSH -L).
- Health check: `curl http://localhost:8000/health` → `{"status":"ok"}`.

### 5. Frontend on the GPU instance

In a **second terminal** on the same GPU instance:

```bash
cd /workspace/ironsite/code
# Use Bun if installed, otherwise npm
bun install   # or: npm install
bun dev       # or: npm run dev
```

- Frontend: **http://localhost:3000**.  
- If you used SSH `-L 3000:localhost:3000`, open **http://localhost:3000** in your Mac browser.

The dashboard uses `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). If the frontend and backend are on the same host and you’re using port forwarding, leave it unset.

---

## Option B: Backend on GPU, frontend on your Mac

Run the backend on the GPU instance, and the dashboard locally, pointing at the GPU backend.

### 1. On the GPU instance

Do **Option A steps 1–4** (SSH, project, models, backend setup and run).  
Ensure the backend is listening on `0.0.0.0:8000` and that your SSH forwards 8000:

```bash
ssh -L 8000:localhost:8000 YOUR_USER@YOUR_GPU_HOST
# then start backend as in Option A step 4
```

### 2. On your Mac

```bash
cd ironsite-hackathon-project-safety_assistant/code
npm install   # or bun install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev   # or bun dev
```

Open **http://localhost:3000**. The app will call the backend at `http://localhost:8000` via the SSH tunnel.

---

## Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for the frontend |
| `SAM_MODEL_PATH` | `/workspace/models/sam3.pt` | Path to SAM3 weights (e.g. local: `backend/models/sam3.pt`) |
| `VLM_MODEL_ID` | `/workspace/models/qwen3-vl` | Qwen3-VL model path or HuggingFace ID |
| `VLM_ADAPTER_PATH` | `/workspace/models/adapter1` | Optional LoRA adapter path |
| `CUDA_VISIBLE_DEVICES` | - | Comma-separated GPU IDs (e.g. `0,1,3`) |
| `JOBS_DIR` | `backend/jobs` | Directory for job outputs |
| `DATABASE_URL` | SQLite under `backend/data/` | DB URL (default is fine for single instance) |

---

## Quick reference

| Step | Command |
|------|--------|
| Backend (GPU) | `cd backend && source venv/bin/activate && CUDA_VISIBLE_DEVICES=0 uvicorn main:app --host 0.0.0.0 --port 8000` |
| Frontend | `cd code && (bun dev \|\| npm run dev)` |
| Health check | `curl http://localhost:8000/health` |
| Frontend URL | http://localhost:3000 |

---

## Troubleshooting

- **Backend won’t start**: Check that `backend/models/last.pt` exists and that SAM/VLM paths exist (or set `SAM_MODEL_PATH`, `VLM_MODEL_ID`, `VLM_ADAPTER_PATH`).
- **“Connection refused” from frontend**: Ensure the backend is running and that `NEXT_PUBLIC_API_URL` matches how you reach it (e.g. `http://localhost:8000` when using SSH -L).
- **OOM / CUDA errors**: Reduce `CUDA_VISIBLE_DEVICES` to one GPU or lower batch/resolution in the pipeline.
- **decord**: Required only for certain video features. On macOS ARM it’s commented out in `requirements.txt`; on Linux GPU run `pip install decord` inside the backend venv if needed.
