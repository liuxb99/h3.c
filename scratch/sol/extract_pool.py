lines = open("h3_shaders.metal").read().splitlines()
pre = lines[:130]  # includes + bf16 helpers
i0 = next(i for i, l in enumerate(lines) if "struct sol_pool_args" in l)
i1 = next(i for i in range(i0, len(lines)) if lines[i] == "}")  # end of pool kernel
body = lines[i0:i1+1]
out = "\n".join(pre + body) + "\n"
open("scratch/sol/pool_only.metal", "w").write(out)
print("wrote", len(out), "bytes; kernel:", any("h3_sol_block_pool_bf16" in l for l in body),
      "; helpers:", any("h3_bf16_to_f32" in l for l in pre))
