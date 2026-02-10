#!/usr/bin/env python3
"""Patch openwakeword train.py to emit METRIC: lines for MLflow tracking."""

import sys

train_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/openwakeword/openwakeword/train.py"

with open(train_path) as f:
    code = f.read()

# Fix: call model.eval() before ONNX export to avoid training-mode artifacts
patches = [
    ('torch.onnx.export(model_to_save.to("cpu"), torch.rand(self.input_shape)[None, ],',
     'model_to_save.eval()\n        torch.onnx.export(model_to_save.to("cpu"), torch.rand(self.input_shape)[None, ],'),
    ('self.history["loss"].append(loss.detach().cpu().numpy())',
     'self.history["loss"].append(loss.detach().cpu().numpy())\n                    _m = self.history["loss"][-1]; print(f"METRIC:loss={_m}:step={step_ndx}", flush=True)'),
    ('self.history["recall"].append(self.recall(accumulated_predictions, accumulated_labels).detach().cpu().numpy())',
     'self.history["recall"].append(self.recall(accumulated_predictions, accumulated_labels).detach().cpu().numpy())\n                    _m = self.history["recall"][-1]; print(f"METRIC:recall={_m}:step={step_ndx}", flush=True)'),
    ('self.history["val_accuracy"].append(val_acc.detach().cpu().numpy())',
     'self.history["val_accuracy"].append(val_acc.detach().cpu().numpy())\n                _m = self.history["val_accuracy"][-1]; print(f"METRIC:val_accuracy={_m}:step={step_ndx}", flush=True)'),
    ('self.history["val_recall"].append(val_recall)',
     'self.history["val_recall"].append(val_recall)\n                _m = self.history["val_recall"][-1]; print(f"METRIC:val_recall={_m}:step={step_ndx}", flush=True)'),
    ('self.history["val_fp_per_hr"].append(val_fp_per_hr)',
     'self.history["val_fp_per_hr"].append(val_fp_per_hr)\n                _m = self.history["val_fp_per_hr"][-1]; print(f"METRIC:val_fp_per_hr={_m}:step={step_ndx}", flush=True)'),
    ('self.history["val_n_fp"].append(val_fp.detach().cpu().numpy())',
     'self.history["val_n_fp"].append(val_fp.detach().cpu().numpy())\n                _m = self.history["val_n_fp"][-1]; print(f"METRIC:val_n_fp={_m}:step={step_ndx}", flush=True)'),
]

applied = 0
for old, new in patches:
    if old in code:
        code = code.replace(old, new, 1)
        applied += 1

with open(train_path, "w") as f:
    f.write(code)

print(f"Patched {applied}/{len(patches)} metric lines in {train_path}")
