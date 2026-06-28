# Experiments & Run Book

End-to-end guide for actually running the dino-ml pipelines and the ML experiments the
roadmap calls for. This is the part that can only be done on a real machine: a GPU plus a live
Chrome Dino window. Unit tests and CI cover the pure logic; everything here is empirical.

Prerequisites:
- A GPU machine (training assumes CUDA; it falls back to CPU but will be slow).
- Chrome with the Dino game open (`chrome://dino`), kept on-screen and focused.
- `pip install -r requirements-dev.txt`.
- All commands run from the repo root as modules.

Config knobs live in `src/ml/dino/config.py` (`DinoEnvConfig`): `obs_size`, `n_stack`,
`frame_skip`, `reward_alive`, `reward_death`, `dead_threshold`, ROI, keys, timing.

> Changing `obs_size` or `n_stack` invalidates any recorded dataset and saved checkpoint
> (shape mismatch). Re-record and retrain after such a change.

---

## Phase 0 — Calibrate and verify capture

1. Read the Dino window corner coordinates:
   ```bash
   python -m src.mouse_pos
   ```
   Move the cursor to the top-left and bottom-right of the play area; note the numbers.
2. Set `roi_left/roi_top/roi_width/roi_height` in `config.py` accordingly.
3. Confirm the crop is correct:
   ```bash
   python -m src.test_pos      # writes debug_raw.png and debug_obs.png
   ```
   `debug_obs.png` should clearly show the dino and obstacles. If not, fix the ROI.
4. Tune death detection: with the game over, `detect_dead` should fire. If it mis-triggers,
   adjust `dead_threshold` (default 140; higher = stricter). Inspect the center crop brightness.

**Done when:** the 84x84 observation looks right and game-over is detected reliably.

---

## Phase 1 — Supervised pipeline (behavior cloning)

1. Record human play (play well; variety matters):
   ```bash
   python -m src.ml.sl.recorder      # Ctrl+C to stop -> dino_sl_dataset.npz
   ```
   Tuning in `RecorderCfg` (`recorder.py`): `noop_keep_prob` (how much idle to keep),
   `label_shift_frames` (compensates human reaction lag).
2. Inspect class balance:
   ```bash
   python -m src.ml.sl.report
   ```
   Expect NOOP to dominate; ensure JUMP/DUCK have enough samples (re-record if a class is ~0).
3. Train (fine-tunes if `dino_sl_cnn.pt` already exists; early-stops on val accuracy):
   ```bash
   python -m src.ml.sl.train         # -> dino_sl_cnn.pt
   ```
4. Run it live:
   ```bash
   python -m src.ml.sl.play
   ```
   If it under-jumps, lower `JUMP_TH` in `play.py` (0.70 -> 0.60); if it spams jumps, raise the
   cooldown. DUCK uses hysteresis (`DUCK_ON_TH` / `DUCK_OFF_TH`).

**Validation (plan M5):** record a confusion matrix and compare validation accuracy against
actual in-game survival time — they often disagree (distribution shift); trust survival time.

---

## Phase 2 — Reinforcement pipeline

PPO (on-policy):
```bash
python -m src.ml.rl.train             # -> ppo_dino_final.zip (+ checkpoints/)
tensorboard --logdir ./ppo_dino_tensorboard/
```
DQN (off-policy, replays experience — better sample efficiency on a single env):
```bash
python -m src.ml.rl.train_dqn         # -> dqn_dino_final.zip (+ checkpoints_dqn/)
tensorboard --logdir ./dqn_dino_tensorboard/
```
Run a trained policy and aggregate metrics:
```bash
python -m src.ml.rl.play
python -m src.ml.rl.eval --model ppo_dino_final --episodes 20
```
Watch `rollout/ep_rew_mean` and `rollout/ep_len_mean` (survival) in TensorBoard.

---

## Phase 3 — SL -> RL warm start (plan M6)

The hypothesis: starting RL from the SL-pretrained backbone learns faster than from scratch.

1. **Warm:** ensure `dino_sl_cnn.pt` exists, then `python -m src.ml.rl.train`. Confirm the log
   prints `[warmstart] SL backbone 로드 성공`.
2. **Scratch:** temporarily move the checkpoint (`mv dino_sl_cnn.pt _sl.bak`) and train again
   into a separate TensorBoard run; the log prints the random-init fallback message.
3. Overlay both runs' `ep_len_mean` vs timesteps.

**Validation:** warm should reach a given survival length in fewer timesteps. If not, the SL
features may not transfer — note it; that is itself a real result.

---

## Phase 4 — DQN vs PPO sample efficiency (plan M7)

Run Phase 2's PPO and DQN to the same `total_timesteps`, both warm-started identically, and
compare survival vs timesteps in TensorBoard. On a single real-screen env, DQN's replay reuse
typically wins early. Record which reaches a target survival first.

---

## Phase 5 — Reward shaping (plan M7)

Edit `reward_alive` / `reward_death` in `config.py` and retrain to study behavior:
- Large `reward_death` magnitude -> cautious, sometimes jump-spamming policies.
- Small alive reward vs death penalty changes the survival/risk trade-off.
Keep each variant in its own TensorBoard run and compare. (Optionally extend `reward.py` with a
progress term once score OCR is wired — see Phase 6.)

---

## Phase 6 — Score OCR templates (plan M4)

`src/ml/dino/score.py` implements template-matching OCR but ships **without** real digit
templates (the font is the game's). To enable it:
1. Capture the score strip from a frame (`signals.score_crop(gray)` gives the top-right region).
2. Crop each digit 0-9 once and store as small 2D arrays -> `templates = {0: arr0, ... 9: arr9}`.
3. Call `read_score(gray, templates)` in your loop to log/score episodes.
The matching/segmentation algorithm is already unit-tested; only the templates are missing.

---

## Phase 7 — Real-time async player (plan M3 / M9)

```bash
python -m src.ml.sl.play_async
```
Capture runs on a thread; inference always consumes the latest frame. On Ctrl+C it prints
`LatencyTracker` stats and an FPS estimate. Compare against the synchronous `sl.play` to confirm
the loop holds your target FPS (30fps -> ~33ms budget). Tune `frame_skip` in `config.py` if the
loop can't keep up.

---

## Milestone validation checklist

| Milestone | How to validate here |
|---|---|
| M1 frame stacking | Phase 1/2 — agent reacts to obstacle speed (jump timing improves) |
| M2 ROI auto-detect | Phase 0 — `roi.resolve_roi` finds the Chrome window on your machine |
| M3 frame-skip + latency | Phase 7 — FPS held; latency summary within budget |
| M4 death/score signals | Phase 0 (death) + Phase 6 (score OCR templates) |
| M5 SL data report/eval | Phase 1 — confusion matrix vs survival time |
| M6 SL->RL warm start | Phase 3 — warm vs scratch survival curves |
| M7 DQN + reward shaping | Phase 4 + Phase 5 |
| M8 eval harness | Phase 2 — `rl.eval` aggregate report; save run configs |
| M9 async pipeline | Phase 7 — threaded capture holds real-time |

## Troubleshooting

- Nothing happens / clicks miss: ROI is wrong (Phase 0).
- Agent never jumps: lower `JUMP_TH` (SL) or check warm start / reward (RL).
- False game-overs: raise `dead_threshold`.
- RL trains but never improves: verify the env returns sane rewards and `detect_dead` is correct;
  a single noisy death signal corrupts learning.
