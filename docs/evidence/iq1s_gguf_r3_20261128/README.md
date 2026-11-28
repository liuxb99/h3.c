# R3: IQ1_S CPU reference dequant vs sd.cpp golden — PASS

208/208 IQ1_S tensors, 78,274,560 blocks, 20,038,287,360 elements, all tail blocks included.
golden = dequantize_row_iq1_s (stable-diffusion.cpp @e239e4b, upstream pin 13b9d92), linked from build/ggml/src/libggml-base.a.
harness = scripts/iq1s_cpu_ref.c + vendored iq1s_grid_ref.h (2048-entry codebook, same pin).
RESULT: global max/mean abs diff = 0 (bit-exact), bad_tensors=0, runtime 131s, read-only pread, RSS < 1 GB.
NOT_TESTED: Metal/GPU dequant path (R4), non-IQ1_S tensor dequant, safetensors-vs-gguf weight parity.
See parity_full.json (per-tensor) + run_full.log.
