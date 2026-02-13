```
pip install -r requirements.txt

# 프레임
python -m src.mouse_pos
python -m src.test_pos

python .\src\hello.py

# 녹화
python -m .src.ml.recorder
# dino_sl_dataset.npz (Supervised Learning용 데이터셋)

# 훈련
python -m src.ml.sl.train
# dino_sl_cnn.pt (CNN 모델) python .\src\ml\train.py


# SL 실행
python -m src.ml.sl.play



# RL
python -m src.ml.rl.policy
python -m src.ml.rl.train
python -m src.ml.rl.play
```


