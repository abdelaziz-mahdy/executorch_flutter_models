"""SPIKE: dynamic XNNPACK Gemma 4 export without custom kv cache.

Mirrors export_mlx_meta.py (optimum's exportable.export(), which is dynamic) but
lowers with XnnpackPartitioner instead of MLXPartitioner. Goal: confirm we get a
DYNAMIC xnnpack forward (parallel prefill works with the stock runner) WITHOUT
--use_custom_kv_cache (which would need custom_ops -> optimized kernels -> the
Xcode 26.5 compile bug). Unquantized (bf16) for the spike; add 8da4w after if the
dynamic hypothesis holds. Also embeds get_eos_ids=[1,106] so the runner stops.
"""
import executorch.exir as exir
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import EdgeCompileConfig
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from optimum.executorch.passes.remove_padding_idx_embedding_pass import (
    RemovePaddingIdxEmbeddingPass,
)
from optimum.exporters.executorch.tasks.causal_lm import load_causal_lm_model

MODEL_ID = "google/gemma-4-E2B-it"
MAX_SEQ_LEN = 512
OUT = "/tmp/gemma4_xnnpack_meta.pte"

# 8da4w linears + 8w embedding = XNNPACK-accelerated (the unquantized bf16 spike
# ran but fell back to portable CPU at ~0.04 tok/s). disable_dynamic_shapes
# defaults False (dynamic) and use_custom_kv_cache defaults False (no custom_ops
# -> no optimized-kernels compile bug). That's the whole point: dynamic + quant
# WITHOUT custom kv cache.
print(f"Loading {MODEL_ID} (8da4w, max_seq_len={MAX_SEQ_LEN})...")
exportable = load_causal_lm_model(
    MODEL_ID,
    revision=None,
    dtype="bfloat16",  # halve RAM vs fp32 default (avoids OOM during quant)
    max_seq_len=MAX_SEQ_LEN,
    qlinear="8da4w",
    # STATIC (decode-only, fixed (1,1) forward). dynamic+8da4w doesn't lower
    # (torchao::*_affine "missing out variants"); static+8da4w does. The native
    # runner auto-detects the static shape and prefills sequentially.
    disable_dynamic_shapes=True,
    # NOTE: NO qembedding — quantizing the embedding breaks a pad_embedding
    # weight-index op in Gemma 4's forward during torch.export (dynamo). The MLX
    # export also quantized only the linears. Embedding stays bf16.
)
print("attn_implementation:", exportable.model.config._attn_implementation)

print("export() ...")
exported_progs = exportable.export()
if len(exported_progs) == 1:
    exported_progs = {"forward": next(iter(exported_progs.values()))}

md = dict(exportable.metadata)
print("optimum metadata:", md)
if "get_eos_id" in md and "get_eos_ids" not in md:
    eos = md.pop("get_eos_id")
    md["get_eos_ids"] = list(eos) if isinstance(eos, (list, tuple)) else [eos]
print("constant_methods:", md)

edge = exir.to_edge_transform_and_lower(
    exported_progs,
    partitioner=[XnnpackPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    constant_methods=md,
    # Fold the padding-idx embedding gather the same way the optimum recipe does.
    transform_passes=[RemovePaddingIdxEmbeddingPass()],
)
prog = edge.to_executorch(
    config=ExecutorchBackendConfig(
        extract_delegate_segments=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        # Required so the torchao 8da4w quant ops are fused into XNNPACK and don't
        # leak as un-loweredtorchao::quantize_affine (-> "Missing out variants").
        do_quant_fusion_and_const_prop=True,
    )
)
with open(OUT, "wb") as f:
    f.write(prog.buffer)
print(f"SAVED {OUT}  size={len(prog.buffer)/1024/1024:.1f} MB")
