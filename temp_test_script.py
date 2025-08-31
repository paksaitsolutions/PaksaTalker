import numpy as np, cv2, wave, struct, math, os
from pathlib import Path

# Prepare tmp dirs
outdir = Path('temp_test')
outdir.mkdir(exist_ok=True)

# Make a simple face image
img = np.zeros((512,512,3), dtype=np.uint8)
cv2.circle(img, (256,256), 200, (180,180,180), -1)
cv2.circle(img, (200,220), 30, (40,40,40), -1)
cv2.circle(img, (312,220), 30, (40,40,40), -1)
cv2.ellipse(img, (256,320), (80,40), 0, 0, 180, (30,30,30), 5)
img_path = str(outdir / 'face.jpg')
cv2.imwrite(img_path, img)

# Make a short WAV audio
sr = 16000
sec = 3
freq = 220.0
wav_path = str(outdir / 'tone.wav')
with wave.open(wav_path, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    for i in range(sr*sec):
        t = i/sr
        val = int(32767*0.25*math.sin(2*math.pi*freq*t))
        wf.writeframes(struct.pack('<h', val))

print('Prepared test image and audio:', img_path, wav_path)

from models.fusion.engine import FusionEngine
eng = FusionEngine()
out_path = 'temp_test/fusion_out.mp4'
print('Generating...')
res = eng.generate(face_image=img_path, audio_path=wav_path, output_path=out_path, emotion='neutral', style='natural', fps=25, resolution='720p', prefer_wav2lip2=False)
print('Done:', res, os.path.exists(res))
