import sys
import os
import cv2

from main import overlay_gdut_badge

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
PREDICTOR = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
BADGE = None
# find badge file (support possible trailing space in filename)
for fname in os.listdir(MODEL_DIR):
    if fname.lower().startswith('gdut_badge') and fname.lower().endswith('.png'):
        BADGE = os.path.join(MODEL_DIR, fname)
        break

if not BADGE:
    print('No badge found in model/. Please place a PNG named gdut_badge.png (or similar).')
    sys.exit(1)

if not os.path.exists(PREDICTOR):
    print('Missing dlib predictor:', PREDICTOR)
    print('Download from: https://github.com/davisking/dlib-models and place in model/')
    sys.exit(1)

if len(sys.argv) < 2:
    print('Usage: python test_ar.py path/to/face_image.jpg')
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print('Input image not found:', img_path)
    sys.exit(1)

img = cv2.imread(img_path)
if img is None:
    print('Failed to load image:', img_path)
    sys.exit(1)

out = overlay_gdut_badge(img)
out_path = os.path.splitext(img_path)[0] + '_ar_out.jpg'
cv2.imwrite(out_path, out)
print('Saved AR output to', out_path)
