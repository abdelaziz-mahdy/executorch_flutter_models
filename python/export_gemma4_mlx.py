"""Export Gemma 4 E2B to an MLX .pte that EMBEDS ET-runner metadata methods.

The stock executorch extension/llm/runner (create_text_llm_runner ->
get_llm_metadata) HARD-REQUIRES a `get_max_seq_len` constant method and reads the
stop set from `get_eos_ids` (PLURAL). The MLX example's custom-components export
path drops constant_methods entirely (only `forward` survives), and even the
optimum path names the stop method `get_eos_id` (SINGULAR). Both make the stock
runner unusable / non-stopping.

This mirrors _export_with_optimum but passes a corrected constant_methods dict so
the resulting pte is fully self-describing: stock runner loads it and stops on
<end_of_turn> with ZERO native workarounds.
"""
import executorch.exir as exir
from executorch.backends.mlx import MLXPartitioner
from executorch.backends.mlx.passes import get_default_passes
from executorch.exir import EdgeCompileConfig
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from optimum.exporters.executorch.tasks.causal_lm import load_causal_lm_model

MODEL_ID = "google/gemma-4-E2B-it"
MAX_SEQ_LEN = 512
OUT = "/tmp/gemma4_mlx_meta.pte"

print(f"Loading {MODEL_ID} via optimum (max_seq_len={MAX_SEQ_LEN})...")
exportable = load_causal_lm_model(
    MODEL_ID, revision=None, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN
)

from executorch.backends.mlx.llm.quantization import quantize_model_

quantize_model_(
    exportable.model,
    qlinear_config="4w",
    qlinear_group_size=None,
    qembedding_config=None,
    qembedding_group_size=None,
    tie_word_embeddings=getattr(
        exportable.model.config, "tie_word_embeddings", False
    ),
)

print("torch.export ...")
exported_progs = exportable.export()
if len(exported_progs) == 1:
    exported_progs = {"forward": next(iter(exported_progs.values()))}

# Build ET-compatible metadata. optimum gives e.g.
#   {'get_eos_id': [1, 106], 'get_max_seq_len': 512, 'use_kv_cache': True,
#    'enable_dynamic_shape': True, 'use_sdpa_with_kv_cache': False, ...}
# The ET runner reads the stop set from 'get_eos_ids' (PLURAL), so rename it.
md = dict(exportable.metadata)
print("optimum metadata:", md)
if "get_eos_id" in md and "get_eos_ids" not in md:
    eos = md.pop("get_eos_id")
    md["get_eos_ids"] = list(eos) if isinstance(eos, (list, tuple)) else [eos]
print("embedded constant_methods:", md)

edge_config = EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
edge_program = exir.to_edge_transform_and_lower(
    exported_progs,
    transform_passes=get_default_passes(),
    partitioner=[MLXPartitioner()],
    compile_config=edge_config,
    constant_methods=md,
)

print("to_executorch ...")
executorch_program = edge_program.to_executorch(
    config=ExecutorchBackendConfig(
        extract_delegate_segments=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    )
)

with open(OUT, "wb") as f:
    f.write(executorch_program.buffer)
print(f"SAVED {OUT}  size={len(executorch_program.buffer)/1024/1024:.1f} MB")
