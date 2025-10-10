# tests/test_imaging_stub.py
from models.imaging_stub import predict_xray_stub
from PIL import Image
import numpy as np

def test_predict_xray_stub(tmp_path):
    path = tmp_path / "img.png"
    arr = (np.ones((256,256))*128).astype('uint8')
    Image.fromarray(arr).save(path)
    probs, sev = predict_xray_stub(str(path))
    assert set(probs.keys()) == {"pneumonia","covid_suspect","normal"}
    assert sev in {"mild","moderate","severe"}