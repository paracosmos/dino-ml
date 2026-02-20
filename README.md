pip install -r requirements.txt

# Frame
python -m src.mouse_pos
python -m src.test_pos

python .\src\hello.py

# Recording
python -m src.ml.recorder
# dino_sl_dataset.npz (Dataset for Supervised Learning)

# Training
python -m src.ml.sl.train
# dino_sl_cnn.pt (CNN model) 
python .\src\ml\train.py

# Run SL
python -m src.ml.sl.play

# RL
python -m src.ml.rl.policy
python -m src.ml.rl.train
python -m src.ml.rl.play
