"""Independent evaluation from NEW held-out photos, without moving the servo.

Put never-trained, never-calibrated photos in:
 data/heldout_target/  (same target plate)
 data/heldout_other/   (different plate IDs)
Run python evaluate.py. This NEVER opens the gate.
"""
from pathlib import Path
import cv2
import joblib
import numpy as np
from plate_core import plate_features

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.png','.jpg','.jpeg','.webp','.bmp'}

def main():
    bundle = joblib.load(ROOT / 'models' / 'plate_model.joblib')
    threshold = float(bundle['threshold'])
    counts = {'TP':0, 'TN':0, 'FP':0, 'FN':0}
    for folder, truth in [('heldout_target', True), ('heldout_other', False)]:
        paths = sorted(p for p in (ROOT / 'data' / folder).iterdir() if p.suffix.lower() in EXTENSIONS)
        if not paths:
            print('No held-out images in', folder, '- add new photos first.')
            continue
        for p in paths:
            im = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                print('Skipped invalid', p)
                continue
            score = float(bundle['model'].decision_function(plate_features(im).reshape(1,-1))[0])
            match = score >= threshold
            key = ('TP' if match else 'FN') if truth else ('FP' if match else 'TN')
            counts[key] += 1
            print(f'{key}: {p.name} (score {score:+.4f}, threshold {threshold:+.4f})')
    print('\nHELD-OUT RESULTS:', counts)
    num_neg = counts['FP'] + counts['TN']
    num_pos = counts['TP'] + counts['FN']
    if num_neg: print('False acceptance rate:', f'{counts["FP"]/num_neg:.1%}', '(lower is better)')
    if num_pos: print('True target recall:', f'{counts["TP"]/num_pos:.1%}')
    print('NO servo commands were sent. Review every false positive before using hardware.')

if __name__ == '__main__':
    main()
