"""Tukar training/model.keras -> docs/model (TF.js graph model)."""
import os, sys, numpy as np
np.object = object  # serasi tensorflowjs lama
import types; sys.modules.setdefault('tensorflow_hub', types.ModuleType('tensorflow_hub'))  # tidak diperlukan
import keras, tensorflow as tf
import tensorflowjs as tfjs
here = os.path.dirname(os.path.abspath(__file__))
m = keras.models.load_model(os.path.join(here, "model.keras"))
sm = os.path.join(here, "_savedmodel")
import shutil; shutil.rmtree(sm, ignore_errors=True); m.export(sm)
tfjs.converters.convert_tf_saved_model(sm, os.path.join(here, "..", "docs", "model"), quantization_dtype_map={"float16": "*"})
print("ok")
