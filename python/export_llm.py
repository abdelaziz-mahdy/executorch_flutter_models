"""Export a Hugging Face causal LM to an ExecuTorch .pte for ExecuTorchLLM.

Produces a self-describing .pte that the stock extension/llm runner loads with
no native workarounds: it embeds `get_max_seq_len` and the stop set as
`get_eos_ids` (plural; optimum names it `get_eos_id`). No custom ops are used,
so the plugin's prebuilt LLM runtimes can run it.

  python export_llm.py --model Qwen/Qwen3-0.6B --backend xnnpack --out qwen3.pte
  python export_llm.py --model Qwen/Qwen3-0.6B --backend mlx --out qwen3_mlx.pte

Backends:
  xnnpack  CPU, all platforms. 8da4w linears, 8-bit embedding.
  mlx      Apple-Silicon GPU, macOS arm64 only. 4-bit linears.
"""
import argparse

import executorch.exir as exir
from executorch.exir import EdgeCompileConfig
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from optimum.exporters.executorch.tasks.causal_lm import load_causal_lm_model


def runner_metadata(exportable, extra_eos):
    """optimum's metadata, with the stop set renamed for the stock runner."""
    md = dict(exportable.metadata)
    eos = md.pop("get_eos_id", None)
    if "get_eos_ids" not in md:
        ids = [] if eos is None else (list(eos) if isinstance(eos, (list, tuple)) else [eos])
        md["get_eos_ids"] = ids
    for token_id in extra_eos:
        if token_id not in md["get_eos_ids"]:
            md["get_eos_ids"].append(token_id)
    return md


def export_xnnpack(args):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from optimum.executorch.passes.remove_padding_idx_embedding_pass import (
        RemovePaddingIdxEmbeddingPass,
    )

    exportable = load_causal_lm_model(
        args.model,
        revision=None,
        dtype="float32",
        max_seq_len=args.max_seq_len,
        qlinear="8da4w",
        qembedding="8w",
        disable_dynamic_shapes=not args.dynamic,
    )
    progs = exportable.export()
    if len(progs) == 1:
        progs = {"forward": next(iter(progs.values()))}
    md = runner_metadata(exportable, args.extra_eos)
    print("metadata:", md)
    edge = exir.to_edge_transform_and_lower(
        progs,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods=md,
        transform_passes=[RemovePaddingIdxEmbeddingPass()],
    )
    return edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            # Fuses the torchao quant ops into XNNPACK; without it they leak
            # as un-lowered torchao::quantize_affine ("Missing out variants").
            do_quant_fusion_and_const_prop=True,
        )
    )


def export_mlx(args):
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.llm.quantization import quantize_model_
    from executorch.backends.mlx.passes import get_default_passes

    exportable = load_causal_lm_model(
        args.model, revision=None, dtype="bfloat16", max_seq_len=args.max_seq_len
    )
    quantize_model_(
        exportable.model,
        qlinear_config="4w",
        qlinear_group_size=None,
        qembedding_config=None,
        qembedding_group_size=None,
        tie_word_embeddings=getattr(exportable.model.config, "tie_word_embeddings", False),
    )
    progs = exportable.export()
    if len(progs) == 1:
        progs = {"forward": next(iter(progs.values()))}
    md = runner_metadata(exportable, args.extra_eos)
    print("metadata:", md)
    edge = exir.to_edge_transform_and_lower(
        progs,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods=md,
    )
    return edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="Hugging Face model id")
    p.add_argument("--backend", required=True, choices=["xnnpack", "mlx"])
    p.add_argument("--out", required=True)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument(
        "--static",
        dest="dynamic",
        action="store_false",
        help="XNNPACK: export a decode-only (1,1) forward (sequential prefill)",
    )
    p.add_argument(
        "--extra-eos",
        type=int,
        nargs="*",
        default=[],
        help="Extra stop token ids (e.g. the end-of-turn token)",
    )
    args = p.parse_args()

    prog = export_xnnpack(args) if args.backend == "xnnpack" else export_mlx(args)
    with open(args.out, "wb") as f:
        f.write(prog.buffer)
    print(f"SAVED {args.out}  {len(prog.buffer) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
