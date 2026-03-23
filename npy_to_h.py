import numpy as np
import sys
from pathlib import Path

npy_path = Path(sys.argv[1])
var_name = npy_path.stem

x = np.load(npy_path)

assert x.shape == (32, 32, 7)

flat = x.reshape(-1)

out_path = npy_path.with_suffix(".h")

with open(out_path, "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write(f"const int8_t {var_name}[{len(flat)}] = {{\n")

    for i, v in enumerate(flat):
        if i % 16 == 0:
            f.write("    ")
        f.write(f"{int(v)}, ")
        if i % 16 == 15:
            f.write("\n")

    f.write("\n};\n")

print("Saved:", out_path)