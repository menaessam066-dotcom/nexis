"""Launch an already-trained local FastAPI app in a browser."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
import subprocess
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parent

def main():
    if not (ROOT / 'models' / 'plate_model.joblib').exists():
        print('Pre-trained model not found. Train it using python train.py', flush=True)
        raise SystemExit(1)
    url = 'http://127.0.0.1:8000'
    print('\nPLATEGATE AI STUDIO', flush=True)
    print('Dashboard URL:', url, flush=True)
    print('Press Ctrl+C to stop the server.\n', flush=True)
    p = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', '8000'], cwd=ROOT)
    for _ in range(70):
        if p.poll() is not None:
            raise RuntimeError('Server failed: port 8000 may already be used. Stop any other server and retry.')
        try:
            with urlopen(url + '/api/status', timeout=.4):
                webbrowser.open(url)
                break
        except Exception:
            time.sleep(.15)
    else:
        p.terminate()
        raise RuntimeError('Local server did not respond.')
    print('Dashboard running. Arduino must be connected in the Arduino & setup tab to move a real servo.', flush=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()

if __name__ == '__main__':
    main()
