# HLS (Vitis HLS) sources

Original hardware work from 2025, kept as found. None of it has been
synthesised or simulated during the reconstruction (no Vitis HLS install).

## `image_derivative/`

3x3 Sobel `dx`/`dy` kernel with AXI-Stream I/O for the KV260 part
(`xck26-sfvc784-2LV-c`), plus a C-simulation testbench.

```bash
cd hls/image_derivative
make csim      # C simulation against the golden reference in tb/testbench.cpp
make csynth    # synthesis report
```

Known issue (from reading the code, not from a run): the kernel emits the
derivative for the window centred on pixel `(x-1, y-2)` (line buffers hold the
three rows *before* the current one), while the testbench stores it at
`(x, y)` and compares against a golden reference centred there. Expect
`make csim` to report mismatches until the index mapping is aligned.

## `ports/`

C++ ports of the Python pipeline stages written as a starting point for
acceleration. They are drafts:

| File | Mirrors | State |
|---|---|---|
| `ego_motion.hpp` | `collision_avoidance/ego_motion.py` | calls Vitis Vision kernels inside a pipelined loop and uses mean-absolute rather than squared error; not synthesisable as written |
| `ttc_estimation.hpp` | `collision_avoidance/ttc.py` | divergence + flow-magnitude estimates only (no looming term); uses ROI slicing of `xf::cv::Mat`, which Vitis Vision does not provide |
| `object_tracker.hpp` | `collision_avoidance/tracking.py` | host C++ (OpenCV), needs a Hungarian solver header that is not included; `predict_new_location` looks up the wrong track |
| `collision_checker.hpp` | ROI / threshold check in `pipeline.py` | host C++, complete |

Constants (ROI, thresholds, focal length, fusion weights) match the Python
defaults in `collision_avoidance/config.py`.
