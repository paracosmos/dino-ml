# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## What this is

An ML bot that plays the **Chrome Dino game** by capturing the live screen and simulating real keyboard/mouse input — it does not run a simulated game. Two independent approaches share the same CNN backbone:

- **SL (supervised / behavior cloning):** record a human playing, then train a CNN classifier to imitate the recorded action labels.
- **RL (reinforcement learning):** a Gymnasium `DinoEnv` driven by PPO from `stable_baselines3`.

Comments throughout the code are in Korean.

## Running

All entry points are modules run from the **repo root** (note the `src.` package prefix is mandatory):

```bash
pip install -r requirements.txt

# Calibrate the screen ROI first (see "Screen ROI" below)
python -m src.mouse_pos     # prints live mouse coordinates
python -m src.test_pos      # captures one ROI frame -> debug_raw.png / debug_obs.png

# Supervised pipeline
python -m src.ml.sl.recorder   # record human play -> dino_sl_dataset.npz
python -m src.ml.sl.train      # train -> saves backbone to dino_sl_cnn.pt
python -m src.ml.sl.play       # run the trained SL model live

# RL pipeline
python -m src.ml.rl.train      # PPO -> checkpoints/, best_model/, ppo_dino_final.zip
python -m src.ml.rl.play       # run a trained PPO model live
```

The `train`/`play` scripts run on CUDA when available and fall back to CPU otherwise.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                 # runs tests/ (config, action spec, preprocessing, model build/forward)
python -m compileall src   # byte-compile sanity check
```

Tests cover the headless-safe units only — they deliberately avoid importing `dino_env`,
`recorder`, and the live players, which require a display (`pyautogui`/`pynput`) and a real
game window. CI (`.github/workflows/ci.yml`) installs deps and runs `compileall` + `pytest`
on every push/PR.

## Architecture

```
src/ml/dino/     environment + shared game I/O
  config.py        DinoEnvConfig — ROI coords, fps, keys, timing (single source of truth)
  preprocess.py    BGR frame -> 84x84x1 grayscale obs (uint8 for storage, float for inference)
  action_spec.py   DinoAction IntEnum: NOOP=0, JUMP=1, DUCK=2
  dino_env.py      DinoEnv(gym.Env) — capture, act, reward, death detection

src/ml/model/
  cnn_backbone.py      DinoCNNBackbone — the shared Conv stack (this is the canonical one)

src/ml/sl/         recorder.py, train.py, play.py
src/ml/rl/         policy.py (SB3 features extractor), train.py, play.py
```

Key cross-cutting facts that aren't obvious from one file:

- **`DinoEnvConfig` is the contract** between every component. Observation size (84), ROI, fps, and key bindings all come from it. The recorder, the env, and the live players all reconstruct observations the same way via `preprocess.py`, so changing `obs_size` or the preprocessing requires retraining and updating `cnn_backbone.py`'s hardcoded `84` sample shape.

- **Death detection is heuristic, not from the game.** `DinoEnv._is_dead()` checks whether the mean brightness of a center crop exceeds 140 (the game-over screen is bright). Reward is `+1` per surviving step, `-100` on death.

- **SL and RL share the CNN but diverge after it.** `DinoSLModel` puts a classification head on the backbone; the SL pipeline saves the **full `DinoSLModel`** state dict to `dino_sl_cnn.pt`, which is what `sl/play.py` loads back. The RL side wraps the backbone in an SB3 `BaseFeaturesExtractor` (`rl/policy.py`) and `rl/train.py` uses `DinoFeatureExtractor` from there.

- **`DinoCNNBackbone.features_dim`** is computed once from a dummy forward pass and read by both `DinoSLModel` and `rl/policy.py` to size their first linear layer.

## Screen ROI (must calibrate before anything works)

`config.py` hardcodes absolute screen pixel coordinates (`roi_left=3930`, etc.) for one specific multi-monitor setup. On any other machine these are wrong and every capture/click will miss the game. Use `src/mouse_pos.py` to read the corner coordinates of the Dino window (its trailing comments document how the current values were derived) and update `DinoEnvConfig`. `focus_point` is where the env clicks to focus the game before acting.

## Notes for contributors

- **One env only for RL.** `rl/train.py` uses a single `DummyVecEnv` because every env captures and controls the *same* physical screen/keyboard — parallel envs would fight each other. Do not reintroduce `SubprocVecEnv`/`N_ENVS>1` until the capture/control layer is per-instance.

## Notes

- `requirements.txt` is a pinned freeze (UTF-8). `requirements-dev.txt` adds `pytest` on top of it.
- All model artifacts (`*.npz`, `*.pt`, `*.zip`) and `debug_*.png` are gitignored; the repo ships code only.
- Originally developed on Windows (README uses `.\src\...` paths); the package layout works cross-platform when invoked with `python -m`.
