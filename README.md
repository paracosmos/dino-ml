pip install -r requirements.txt

# Frame / ROI calibration
python -m src.mouse_pos
python -m src.test_pos
python -m src.hello

# Recording
python -m src.ml.sl.recorder
# dino_sl_dataset.npz (Dataset for Supervised Learning)

# Training (SL)
python -m src.ml.sl.train
# dino_sl_cnn.pt (model checkpoint)

# Run SL
python -m src.ml.sl.play

# RL
python -m src.ml.rl.train
python -m src.ml.rl.play
