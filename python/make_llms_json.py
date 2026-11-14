"""Write <version>/llms.json, the catalog of hosted LLMs for the example app.

LLM exports are made by hand (export_llm.py) and uploaded to the version's
GitHub release next to the vision models. This lists them, with sizes and
SHA-256 hashes read from the local files, so the example app can download and
verify them. It is a separate file from index.json because the export
workflow rewrites index.json on every run.

  python make_llms_json.py --version 1.6.0 --dir /path/to/exports

The models and their chat formats are declared in LLMS below; every file named
there must exist in --dir.
"""
import argparse
import hashlib
import json
from pathlib import Path

CHATML_QWEN3 = {
    "id": "chatml-qwen3",
    "label": "ChatML (Qwen3)",
    "user": "<|im_start|>user\n{text}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{text}<|im_end|>\n",
    # The empty think block is what Qwen3's own template inserts for
    # enable_thinking=False.
    "generationPrompt": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "stop": ["<|im_end|>", "<|endoftext|>"],
}

QWEN3 = {
    "displayName": "Qwen3 0.6B",
    "license": "Apache-2.0",
    "sourceUrl": "https://huggingface.co/Qwen/Qwen3-0.6B",
    "chatFormat": CHATML_QWEN3,
}

LLMS = [
    {
        **QWEN3,
        "id": "qwen3-0.6b-xnnpack",
        "backend": "xnnpack",
        "files": {
            "model": "qwen3_0.6b_xnnpack.pte",
            "tokenizer": "qwen3_0.6b_tokenizer.json",
        },
    },
    {
        **QWEN3,
        "id": "qwen3-0.6b-mlx",
        "backend": "mlx",
        "files": {
            "model": "qwen3_0.6b_mlx.pte",
            "tokenizer": "qwen3_0.6b_tokenizer.json",
            # From the matching executorch_native xnnpack-mlx-llm prebuilt.
            "metallib": "mlx.metallib",
        },
    },
]


def describe(path: Path, base_url: str) -> dict:
    data = path.read_bytes()
    return {
        "name": path.name,
        "url": f"{base_url}/{path.name}",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--version", required=True, help="ExecuTorch version, e.g. 1.6.0")
    p.add_argument("--dir", required=True, help="Directory holding the exported files")
    p.add_argument("--out", help="Output path (default: ../<version>/llms.json)")
    args = p.parse_args()

    base_url = (
        "https://github.com/abdelaziz-mahdy/executorch_flutter_models"
        f"/releases/download/v{args.version}"
    )
    src = Path(args.dir)
    llms = []
    for entry in LLMS:
        files = {
            role: describe(src / name, base_url)
            for role, name in entry["files"].items()
        }
        llms.append({**entry, "files": files})

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / args.version / "llms.json"
    )
    out.write_text(json.dumps({"version": args.version, "llms": llms}, indent=2) + "\n")
    print(f"Wrote {out} ({len(llms)} LLMs)")


if __name__ == "__main__":
    main()
