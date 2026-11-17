"""Local-only plate verifier + Arduino servo demo dashboard."""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from pathlib import Path
from threading import Lock
from datetime import datetime
import json
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from plate_core import decode_image, plate_features
from gate import controller

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'plate_model.joblib'
MAX_BYTES = 8 * 1024 * 1024
VALID_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
app = FastAPI(title='PlateGate AI Studio', version='3.0')
recent = deque(maxlen=10)
predict_mutex = Lock()


@lru_cache(maxsize=2)
def _load_model(modified_ns: int):
    # Only load models generated LOCALLY by our own trusted training script.
    return joblib.load(MODEL_PATH)


def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('Model missing — run python train.py first.')
    # Model is reloaded automatically if retrain_windows.bat replaces it.
    return _load_model(MODEL_PATH.stat().st_mtime_ns)


@app.get('/', response_class=HTMLResponse)
def home():
    return (ROOT / 'static' / 'index.html').read_text(encoding='utf-8')


@app.get('/static/{asset_name}')
def asset(asset_name: str):
    if asset_name not in ('style.css', 'app.js'):
        raise HTTPException(404, 'Unknown asset')
    return FileResponse(ROOT / 'static' / asset_name)


def counts():
    return {k: len([p for p in (ROOT / 'data' / k).iterdir() if p.suffix.lower() in VALID_EXT]) for k in ('target', 'other')}


@app.get('/api/status')
def status():
    return {'server': 'online', 'model_ready': MODEL_PATH.is_file(), 'dataset': counts(), 'gate': controller.status(), 'history': list(recent)}


@app.get('/api/model')
def model():
    try:
        report = get_model().get('report', {})
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    return report


@app.get('/api/dataset')
def dataset():
    return {k: sorted(p.name for p in (ROOT / 'data' / k).iterdir() if p.suffix.lower() in VALID_EXT) for k in ('target', 'other')}


@app.get('/dataset/{kind}/{filename}')
def dataset_photo(kind: str, filename: str):
    if kind not in ('target', 'other') or filename != Path(filename).name or Path(filename).suffix.lower() not in VALID_EXT:
        raise HTTPException(404, 'Not found')
    p = ROOT / 'data' / kind / filename
    if not p.is_file():
        raise HTTPException(404, 'Not found')
    return FileResponse(p, headers={'Cache-Control': 'private, max-age=3600'})


@app.get('/api/ports')
def ports():
    return {'ports': controller.ports()}


class ConnectRequest(BaseModel):
    port: str


@app.post('/api/connect')
def connect(req: ConnectRequest):
    try:
        return {'gate': controller.connect(req.port)}
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e


@app.post('/api/disconnect')
def disconnect():
    try:
        return {'gate': controller.disconnect()}
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.post('/api/reset-demo')
def reset_demo():
    try:
        return {'gate': controller.reset_error()}
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.post('/api/retry-close')
def retry_close():
    try:
        return {'gate': controller.retry_close()}
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if not predict_mutex.acquire(blocking=False):
        raise HTTPException(409, 'Another image is being analyzed. Try again shortly.')
    try:
        gate_status = controller.status()
        if not gate_status['can_accept_image']:
            problem = gate_status['error'] or gate_status['last_event']
            raise HTTPException(409, f'Gate not ready ({gate_status["state"]}). {problem}')
        if file.content_type and file.content_type not in ('image/jpeg','image/png','image/webp','image/bmp','application/octet-stream'):
            raise HTTPException(415, 'Upload a JPG, PNG, WEBP, or BMP photo.')
        blob = await file.read(MAX_BYTES + 1)
        if len(blob) > MAX_BYTES:
            raise HTTPException(413, 'Max photo size is 8 MB.')
        try:
            img = decode_image(blob)
        except (ValueError, Exception) as e:
            raise HTTPException(400, f'Unable to read image: {e}') from e
        try:
            bundle = get_model()
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from e
        score = float(bundle['model'].decision_function(plate_features(img).reshape(1, -1))[0])
        threshold = float(bundle['threshold'])
        is_match = score >= threshold
        gate_triggered = False
        gate_message = 'Different plate — gate remains CLOSED'
        if is_match:
            gate_triggered, gate_message = controller.trigger()
        row = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'filename': (file.filename or 'uploaded_image')[:100],
            'result': 'MATCH' if is_match else 'NOT MATCH',
            'gate_triggered': gate_triggered,
        }
        recent.appendleft(row)
        return {
            'match': bool(is_match),
            'result': row['result'],
            'decision_score': round(score, 5),
            'decision_threshold': round(threshold, 5),
            'gate_triggered': gate_triggered,
            'gate_message': gate_message,
            'gate': controller.status(),
            'note': 'Synthetic-data demo; image must be tightly cropped to a single plate. Score is not a probability.',
        }
    finally:
        predict_mutex.release()


@app.get('/samples/{sample_name}')
def sample_photo(sample_name: str):
    if sample_name not in ('test_known.jpg', 'test_unknown.jpg'):
        raise HTTPException(404, 'Unknown sample')
    return FileResponse(ROOT / 'samples' / sample_name)


@app.post('/api/force-close')
def force_close():
    """Best-effort CLOSE over serial, then disconnect; verify position physically."""
    return {'gate': controller.disconnect(force=True),
            'warning': 'CLOSE sent if possible. Confirm the servo is physically closed before reconnecting.'}
