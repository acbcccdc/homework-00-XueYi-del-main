import urllib.request
import bz2
import os
import shutil

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
PREDICTOR_BZ2 = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat.bz2")
PREDICTOR_DAT = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")

def download_and_extract():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if os.path.exists(PREDICTOR_DAT):
        print("Predictor already exists:", PREDICTOR_DAT)
        return

    print("Downloading predictor from", PREDICTOR_URL)
    try:
        urllib.request.urlretrieve(PREDICTOR_URL, PREDICTOR_BZ2)
        print("Download complete. Extracting...")
        
        with bz2.BZ2File(PREDICTOR_BZ2, 'rb') as f_in:
            with open(PREDICTOR_DAT, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print("Extraction complete:", PREDICTOR_DAT)
        os.remove(PREDICTOR_BZ2) # Cleanup
    except Exception as e:
        print("Error downloading/extracting:", e)

if __name__ == "__main__":
    download_and_extract()
