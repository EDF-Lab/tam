import subprocess
import sys

for effect in ["categorical", "chebychev", "fourier", "linear", "linear_tree", "nept_neural", "pid",
               "rbf", "spline", "tensor_interaction", "tree", "wavelet", "pikl_physics"]:
    subprocess.run([sys.executable, f"slides/{effect}_slide.py"], check=True)