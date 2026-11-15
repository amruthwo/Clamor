#!/usr/bin/env python3
"""Write an f32 copy of a GGUF (every tensor dequantized; metadata copied).

Usage: dequant_gguf.py IN.gguf OUT.gguf
Used to check the engine against tools/ref_forward.py without quantization noise: with f32
weights the two must agree to ~1e-5, so any larger difference is a real bug.
"""
import sys
import numpy as np
import gguf

src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = gguf.GGUFWriter(dst, arch)
for k, f in r.fields.items():
    if k.startswith("GGUF.") or k in ("general.architecture", "general.file_type"):
        continue
    vtype = f.types[0]
    sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(k, f.contents(), vtype, sub)
w.add_file_type(gguf.LlamaFileType.ALL_F32)
for t in r.tensors:
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        x = np.array(t.data, dtype=np.float32)
    else:
        x = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
    w.add_tensor(t.name, x.reshape(tuple(reversed(t.shape.tolist()))))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {dst}")
