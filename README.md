# dino-ml

Machine-learning bot that plays the **Chrome Dino game** by capturing the live screen
and sending real keyboard/mouse input. Two approaches share one CNN backbone:

- **SL (behavior cloning)** — record human play, train a CNN to imitate the actions.
- **RL (PPO / DQN)** — a Gymnasium environment trained with stable-baselines3.

Observations are frame-stacked grayscale crops (`n_stack` frames) so the agent can perceive
motion. The SL backbone can warm-start RL.

- **[AGENTS.md](./AGENTS.md)** — architecture and contributor notes.
- **[docs/EXPERIMENTS.md](./docs/EXPERIMENTS.md)** — step-by-step run book for the full
  record -> train -> evaluate workflow and the ML experiments (warm-start, DQN vs PPO,
  reward shaping, latency).

## Setup

```bash
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest (for tests)
```

All commands run as modules from the repo root.

## Calibrate the screen ROI (do this first)

`DinoEnvConfig` hardcodes absolute screen coordinates for one monitor setup. On any other
machine they are wrong, so calibrate before anything else:

```bash
python -m src.mouse_pos     # print live mouse coordinates to read the Dino window corners
python -m src.test_pos      # capture one ROI frame -> debug_raw.png / debug_obs.png
```

## Supervised pipeline

```bash
python -m src.ml.sl.recorder   # record human play -> dino_sl_dataset.npz
python -m src.ml.sl.report     # inspect the recorded class distribution
python -m src.ml.sl.train      # train -> dino_sl_cnn.pt
python -m src.ml.sl.play       # run the trained model live
python -m src.ml.sl.play_async # same, with threaded capture (latest-frame, real-time)
```

## Reinforcement pipeline

```bash
python -m src.ml.rl.train       # PPO  -> ppo_dino_final.zip (+ checkpoints/)
python -m src.ml.rl.train_dqn   # DQN  -> dqn_dino_final.zip (off-policy, more sample-efficient)
python -m src.ml.rl.play        # run a trained PPO model live
python -m src.ml.rl.eval --model ppo_dino_final --episodes 10   # aggregate return/survival
```

If `dino_sl_cnn.pt` is present, RL training warm-starts its CNN backbone from it.

## Tests

```bash
pytest                 # pure logic: preprocessing, models, signals, reward, OCR, utils
python -m compileall src
```

CI (`.github/workflows/ci.yml`) runs these on every push/PR. Modules that drive the real game
(env, recorder, players, RL training/eval) need a display and the live game, so they are only
byte-compiled in CI; validate them by running against the actual Dino window.

## Notes

- Changing `obs_size` or `n_stack` invalidates recorded datasets and saved checkpoints
  (shape mismatch) — re-record and retrain.
- Model artifacts (`*.npz`, `*.pt`, `*.zip`) and debug images are gitignored.
