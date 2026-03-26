# AGENTS.md

## Cursor Cloud specific instructions

This is a Python desktop application (YOLO-based game aim-assist with Bezier curve mouse simulator). See `README.md` and `README_GUI.md` for full documentation.

### Services

| Service | How to run | Notes |
|---|---|---|
| GUI (Bezier curve simulator) | `DISPLAY=:1 python3 start_gui.py` | Tkinter GUI; requires X display |
| Main (YOLO detection loop) | `DISPLAY=:1 python3 main.py` | Requires ONNX model; see config note below |
| Tests (stdlib-only) | `python3 simple_test.py` | No display needed |
| Tests (full suite) | `DISPLAY=:99 python3 test_robustness.py` | Needs Xvfb on :99; uses `unittest` |

### Key caveats

- **Display requirement**: `pynput` and `pyautogui` fail to import without an X display. Use `DISPLAY=:1` (VNC/Desktop pane) for interactive apps, or `DISPLAY=:99` with Xvfb for headless tests.
- **Xvfb**: Start with `Xvfb :99 -screen 0 1920x1080x24 &` if not already running. The VNC display `:1` is always available.
- **python3-tk**: Required system package for Tkinter (`sudo apt-get install -y python3-tk`).
- **python3-dev**: Required for building `evdev` (dependency of `pynput` on Linux): `sudo apt-get install -y python3-dev`.
- **ONNX model path**: `config.json` has `model_path` set to `yolo/yoloaimonnx/onnxmd/10w320v5.onnx` but models actually live in `onnxmd/`. To run `main.py`, update `config.json` to `"model_path": "onnxmd/10w320v5.onnx"`. Tests mock the model and don't need this fix.
- **CJK font warnings**: Matplotlib emits harmless glyph warnings for Chinese characters when CJK fonts are not installed. This is cosmetic only.

### Lint / Test / Build commands

- **Tests**: `python3 simple_test.py` (stdlib) and `DISPLAY=:99 python3 test_robustness.py` (full)
- **No formal linter** is configured in this project.
- **No build step** — pure Python, run directly.
