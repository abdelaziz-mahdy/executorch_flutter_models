#!/usr/bin/env python3
"""
ExecuTorch Flutter - Main Script

Unified command-line tool for model export and validation.

Usage:
    python main.py                          # Export models (default)
    python main.py export                   # Export models
    python main.py export --mobilenet       # Export MobileNet only
    python main.py export --yolo yolo11n    # Export YOLO11n
    python main.py validate                 # Validate all models
"""

import sys
import argparse
from pathlib import Path

# Export functions
from validate_all_models import ModelValidator

# Validation functions
import torch
import torchvision.models as models


def export_mobilenet(output_dir="..", backends=None):
    """Export MobileNet V3 Small with multiple backend support."""
    print("\n" + "="*70)
    print("  Exporting MobileNet V3 Small")
    print("="*70 + "\n")

    try:
        import torch
        from pathlib import Path
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Default backends: xnnpack for all platforms, coreml/mps for Apple, vulkan for Android
        if backends is None:
            backends = ['xnnpack', 'coreml', 'mps', 'vulkan']

        # Load model
        model = models.mobilenet_v3_small(weights='DEFAULT').eval()
        sample_inputs = (torch.randn(1, 3, 224, 224),)

        # Create exporter
        exporter = ExecuTorchExporter()

        # Filter backends to only available ones
        available_backends = [b for b in backends if exporter.available_backends.get(b, False)]

        if not available_backends:
            print(f"⚠️  No available backends from requested: {backends}")
            print(f"   Available backends: {[k for k, v in exporter.available_backends.items() if v]}")
            return False

        print(f"📦 Exporting to backends: {available_backends}")

        # Output to mobilenet subdirectory
        mobilenet_output_dir = str(Path(output_dir) / "mobilenet")

        # Create export config
        config = ExportConfig(
            model_name='mobilenet_v3_small',
            backends=available_backends,
            output_dir=mobilenet_output_dir,
            quantize=False,
            input_shapes=[[1, 3, 224, 224]],
            input_dtypes=['float32']
        )

        # Export to all backends
        results = exporter.export_model(model, sample_inputs, config)

        # Check success
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if successful:
            print(f"\n✅ Successfully exported {len(successful)}/{len(results)} backends")
            for result in successful:
                print(f"   • {result.backend}: {result.output_path.split('/')[-1]} ({result.file_size_mb:.1f} MB)")

        if failed:
            print(f"\n⚠️  Failed {len(failed)} backend(s):")
            for result in failed:
                print(f"   • {result.backend}: {result.error_message}")

        return len(successful) > 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_yolo(model_name="yolo11n", output_dir="..", backends=None):
    """Export YOLO model with multiple backend support."""
    print("\n" + "="*70)
    print(f"  Exporting {model_name.upper()}")
    print("="*70 + "\n")

    try:
        import torch
        import numpy as np
        from pathlib import Path
        from ultralytics import YOLO
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Default backends: xnnpack for all platforms, coreml/mps for Apple, vulkan for Android
        if backends is None:
            backends = ['xnnpack', 'coreml', 'mps', 'vulkan']

        # Load YOLO model
        model = YOLO(f"{model_name}.pt")

        # Run a dummy prediction to initialize the model
        np_dummy_tensor = np.ones((640, 640, 3))
        model.predict(np_dummy_tensor, imgsz=(640, 640), device="cpu")

        # Get the PyTorch model and put in eval mode
        pt_model = model.model.cpu().eval()

        # Prepare sample inputs (640x640 for YOLO)
        sample_inputs = (torch.randn(1, 3, 640, 640),)

        # Create exporter
        exporter = ExecuTorchExporter()

        # Filter backends to only available ones
        available_backends = [b for b in backends if exporter.available_backends.get(b, False)]

        if not available_backends:
            print(f"⚠️  No available backends from requested: {backends}")
            print(f"   Available backends: {[k for k, v in exporter.available_backends.items() if v]}")
            return False

        print(f"📦 Exporting to backends: {available_backends}")

        # Output to yolo subdirectory
        yolo_output_dir = str(Path(output_dir) / "yolo")

        # Create export config
        config = ExportConfig(
            model_name=model_name,
            backends=available_backends,
            output_dir=yolo_output_dir,
            quantize=False,
            input_shapes=[[1, 3, 640, 640]],
            input_dtypes=['float32']
        )

        # Export to all backends
        results = exporter.export_model(pt_model, sample_inputs, config)

        # Check success
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if successful:
            print(f"\n✅ Successfully exported {len(successful)}/{len(results)} backends")
            for result in successful:
                print(f"   • {result.backend}: {result.output_path.split('/')[-1]} ({result.file_size_mb:.1f} MB)")

        if failed:
            print(f"\n⚠️  Failed {len(failed)} backend(s):")
            for result in failed:
                print(f"   • {result.backend}: {result.error_message}")

        # Clean up downloaded model files (handles both .pt and variant names like yolov5nu.pt)
        for pt_file in Path.cwd().glob("*.pt"):
            if pt_file.stem.startswith(model_name):
                pt_file.unlink()
                print(f"   Cleaned up: {pt_file.name}")

        return len(successful) > 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_movenet(model_variant="lightning", output_dir="..", backends=None):
    """Export MoveNet pose estimation model.

    MoveNet is a pose estimation model from TensorFlow Hub.
    Supports 'lightning' (faster, 192x192) and 'thunder' (more accurate, 256x256).

    Flow: TensorFlow Hub → ONNX → PyTorch → ExecuTorch
    """
    print("\n" + "="*70)
    print(f"  Exporting MoveNet {model_variant.upper()}")
    print("="*70 + "\n")

    try:
        import torch
        import numpy as np
        from pathlib import Path
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Validate variant
        if model_variant not in ["lightning", "thunder"]:
            print(f"❌ Invalid variant: {model_variant}. Use 'lightning' or 'thunder'")
            return False

        # Input size depends on variant
        input_size = 192 if model_variant == "lightning" else 256

        print(f"📦 Loading MoveNet {model_variant} (input size: {input_size}x{input_size})...")

        # Default backends
        if backends is None:
            backends = ['xnnpack', 'coreml', 'mps', 'vulkan']

        # Try to load from TensorFlow Hub and convert to ONNX, then to PyTorch
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            import tf2onnx
            import onnx
            from onnx2torch import convert as onnx_to_torch

            # Load from TensorFlow Hub
            model_url = f"https://tfhub.dev/google/movenet/singlepose/{model_variant}/4"
            print(f"   Loading from TensorFlow Hub: {model_url}")

            # Create a concrete function from the TF Hub model
            module = hub.load(model_url)
            model_fn = module.signatures['serving_default']

            # Create input spec for tf2onnx
            input_spec = tf.TensorSpec([1, input_size, input_size, 3], tf.int32, name='input')

            # Convert to ONNX
            print("   Converting to ONNX...")
            temp_onnx_path = Path(output_dir) / "movenet" / f"movenet_{model_variant}_temp.onnx"
            temp_onnx_path.parent.mkdir(parents=True, exist_ok=True)

            model_proto, _ = tf2onnx.convert.from_function(
                model_fn,
                input_signature=[input_spec],
                opset=13,
                output_path=str(temp_onnx_path)
            )

            # Load ONNX and convert to PyTorch
            print("   Converting ONNX to PyTorch...")
            onnx_model = onnx.load(str(temp_onnx_path))
            pt_model = onnx_to_torch(onnx_model).eval()

            # Clean up temp ONNX file
            temp_onnx_path.unlink()

        except ImportError as e:
            print(f"❌ Missing dependencies for TensorFlow conversion: {e}")
            print(f"\n💡 Install required packages:")
            print(f"   pip install tensorflow tensorflow_hub tf2onnx onnx onnx2torch")
            return False

        # Create exporter
        exporter = ExecuTorchExporter()

        # Filter backends to only available ones
        available_backends = [b for b in backends if exporter.available_backends.get(b, False)]

        if not available_backends:
            print(f"⚠️  No available backends from requested: {backends}")
            print(f"   Available backends: {[k for k, v in exporter.available_backends.items() if v]}")
            return False

        print(f"📦 Exporting to backends: {available_backends}")

        # Prepare sample inputs (NCHW format for PyTorch)
        sample_inputs = (torch.randn(1, 3, input_size, input_size),)

        # Output to movenet subdirectory
        movenet_output_dir = str(Path(output_dir) / "movenet")

        # Create export config
        config = ExportConfig(
            model_name=f'movenet_{model_variant}',
            backends=available_backends,
            output_dir=movenet_output_dir,
            quantize=False,
            input_shapes=[[1, 3, input_size, input_size]],
            input_dtypes=['float32']
        )

        # Export to all backends
        results = exporter.export_model(pt_model, sample_inputs, config)

        # Check success
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if successful:
            print(f"\n✅ Successfully exported {len(successful)}/{len(results)} backends")
            for result in successful:
                print(f"   • {result.backend}: {result.output_path.split('/')[-1]} ({result.file_size_mb:.1f} MB)")

        if failed:
            print(f"\n⚠️  Failed {len(failed)} backend(s):")
            for result in failed:
                print(f"   • {result.backend}: {result.error_message}")

        return len(successful) > 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_blazeface(output_dir="..", backends=None):
    """Export BlazeFace face detection model.

    BlazeFace is a lightweight face detection model from MediaPipe.
    Uses ONNX model from community sources.

    Flow: ONNX → PyTorch → ExecuTorch
    Input size: 128x128
    """
    print("\n" + "="*70)
    print("  Exporting BlazeFace")
    print("="*70 + "\n")

    try:
        import torch
        import urllib.request
        from pathlib import Path
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Default backends
        if backends is None:
            backends = ['xnnpack', 'coreml', 'mps', 'vulkan']

        input_size = 128

        print(f"📦 Loading BlazeFace (input size: {input_size}x{input_size})...")

        try:
            import onnx
            from onnx2torch import convert as onnx_to_torch

            # Download BlazeFace ONNX model from ONNX Model Zoo or community
            # Using the face_detection_front model from MediaPipe via PINTO's conversion
            onnx_url = "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/030_BlazeFace/saved_model/model_float32.onnx"

            temp_onnx_path = Path(output_dir) / "blazeface" / "blazeface_temp.onnx"
            temp_onnx_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"   Downloading BlazeFace ONNX model...")
            urllib.request.urlretrieve(onnx_url, str(temp_onnx_path))

            # Load ONNX and convert to PyTorch
            print("   Converting ONNX to PyTorch...")
            onnx_model = onnx.load(str(temp_onnx_path))
            pt_model = onnx_to_torch(onnx_model).eval()

            # Clean up temp ONNX file
            temp_onnx_path.unlink()

        except ImportError as e:
            print(f"❌ Missing dependencies for ONNX conversion: {e}")
            print(f"\n💡 Install required packages:")
            print(f"   pip install onnx onnx2torch")
            return False
        except urllib.error.URLError as e:
            print(f"❌ Failed to download BlazeFace model: {e}")
            return False

        # Create exporter
        exporter = ExecuTorchExporter()

        # Filter backends to only available ones
        available_backends = [b for b in backends if exporter.available_backends.get(b, False)]

        if not available_backends:
            print(f"⚠️  No available backends from requested: {backends}")
            print(f"   Available backends: {[k for k, v in exporter.available_backends.items() if v]}")
            return False

        print(f"📦 Exporting to backends: {available_backends}")

        # Prepare sample inputs (NCHW format)
        sample_inputs = (torch.randn(1, 3, input_size, input_size),)

        # Output to blazeface subdirectory
        blazeface_output_dir = str(Path(output_dir) / "blazeface")

        # Create export config
        config = ExportConfig(
            model_name='blazeface',
            backends=available_backends,
            output_dir=blazeface_output_dir,
            quantize=False,
            input_shapes=[[1, 3, input_size, input_size]],
            input_dtypes=['float32']
        )

        # Export to all backends
        results = exporter.export_model(pt_model, sample_inputs, config)

        # Check success
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if successful:
            print(f"\n✅ Successfully exported {len(successful)}/{len(results)} backends")
            for result in successful:
                print(f"   • {result.backend}: {result.output_path.split('/')[-1]} ({result.file_size_mb:.1f} MB)")

        if failed:
            print(f"\n⚠️  Failed {len(failed)} backend(s):")
            for result in failed:
                print(f"   • {result.backend}: {result.error_message}")

        return len(successful) > 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_yolo_pose(model_name="yolo11n-pose", output_dir="..", backends=None):
    """Export YOLO Pose estimation model.

    Uses Ultralytics YOLO pose models for human pose estimation.
    Input size: 640x640

    Supported models: yolo11n-pose, yolo11s-pose, yolov8n-pose, yolov8s-pose
    """
    print("\n" + "="*70)
    print(f"  Exporting {model_name.upper()}")
    print("="*70 + "\n")

    try:
        import torch
        import numpy as np
        from pathlib import Path
        from ultralytics import YOLO
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Default backends
        if backends is None:
            backends = ['xnnpack', 'coreml', 'mps', 'vulkan']

        # Load YOLO Pose model
        print(f"📦 Loading {model_name}...")
        model = YOLO(f"{model_name}.pt")

        # Run a dummy prediction to initialize the model
        np_dummy_tensor = np.ones((640, 640, 3))
        model.predict(np_dummy_tensor, imgsz=(640, 640), device="cpu")

        # Get the PyTorch model and put in eval mode
        pt_model = model.model.cpu().eval()

        # Prepare sample inputs (640x640 for YOLO)
        sample_inputs = (torch.randn(1, 3, 640, 640),)

        # Create exporter
        exporter = ExecuTorchExporter()

        # Filter backends to only available ones
        available_backends = [b for b in backends if exporter.available_backends.get(b, False)]

        if not available_backends:
            print(f"⚠️  No available backends from requested: {backends}")
            print(f"   Available backends: {[k for k, v in exporter.available_backends.items() if v]}")
            return False

        print(f"📦 Exporting to backends: {available_backends}")

        # Output to yolo-pose subdirectory
        yolo_pose_output_dir = str(Path(output_dir) / "yolo-pose")

        # Create export config
        config = ExportConfig(
            model_name=model_name,
            backends=available_backends,
            output_dir=yolo_pose_output_dir,
            quantize=False,
            input_shapes=[[1, 3, 640, 640]],
            input_dtypes=['float32']
        )

        # Export to all backends
        results = exporter.export_model(pt_model, sample_inputs, config)

        # Check success
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if successful:
            print(f"\n✅ Successfully exported {len(successful)}/{len(results)} backends")
            for result in successful:
                print(f"   • {result.backend}: {result.output_path.split('/')[-1]} ({result.file_size_mb:.1f} MB)")

        if failed:
            print(f"\n⚠️  Failed {len(failed)} backend(s):")
            for result in failed:
                print(f"   • {result.backend}: {result.error_message}")

        # Clean up downloaded model files
        for pt_file in Path.cwd().glob("*.pt"):
            if pt_file.stem.startswith(model_name.split("-")[0]):
                pt_file.unlink()
                print(f"   Cleaned up: {pt_file.name}")

        return len(successful) > 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_yolo_face(model_name="yolov8n-face", output_dir="..", backends=None):
    """Export YOLO Face detection model.

    Uses Ultralytics YOLO face models for face detection.
    Input size: 640x640

    Supported models: yolov8n-face, yolov8s-face
    Note: YOLO Face models may require downloading from community repos.
    """
    print("\n" + "="*70)
    print(f"  Exporting {model_name.upper()}")
    print("="*70 + "\n")

    try:
        import torch
        import numpy as np
        import urllib.request
        from pathlib import Path
        from ultralytics import YOLO
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Default backends
        if backends is None:
            backends = ['xnnpack', 'coreml', 'mps', 'vulkan']

        print(f"📦 Loading {model_name}...")

        # YOLO Face models need to be downloaded from community repos
        # Using deepcam-cn/yolov5-face adapted for ultralytics
        model_path = Path.cwd() / f"{model_name}.pt"

        if not model_path.exists():
            # Try to download from a community source
            # The akanametov/yolo-face repo provides YOLOv8 face models
            face_model_urls = {
                "yolov8n-face": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
                "yolov8s-face": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8s-face.pt",
            }

            if model_name in face_model_urls:
                print(f"   Downloading {model_name} from community repo...")
                urllib.request.urlretrieve(face_model_urls[model_name], str(model_path))
            else:
                print(f"❌ Unknown face model: {model_name}")
                print(f"   Supported models: {list(face_model_urls.keys())}")
                return False

        # Load YOLO Face model
        model = YOLO(str(model_path))

        # Run a dummy prediction to initialize the model
        np_dummy_tensor = np.ones((640, 640, 3))
        model.predict(np_dummy_tensor, imgsz=(640, 640), device="cpu")

        # Get the PyTorch model and put in eval mode
        pt_model = model.model.cpu().eval()

        # Prepare sample inputs (640x640 for YOLO)
        sample_inputs = (torch.randn(1, 3, 640, 640),)

        # Create exporter
        exporter = ExecuTorchExporter()

        # Filter backends to only available ones
        available_backends = [b for b in backends if exporter.available_backends.get(b, False)]

        if not available_backends:
            print(f"⚠️  No available backends from requested: {backends}")
            print(f"   Available backends: {[k for k, v in exporter.available_backends.items() if v]}")
            return False

        print(f"📦 Exporting to backends: {available_backends}")

        # Output to yolo-face subdirectory
        yolo_face_output_dir = str(Path(output_dir) / "yolo-face")

        # Create export config
        config = ExportConfig(
            model_name=model_name,
            backends=available_backends,
            output_dir=yolo_face_output_dir,
            quantize=False,
            input_shapes=[[1, 3, 640, 640]],
            input_dtypes=['float32']
        )

        # Export to all backends
        results = exporter.export_model(pt_model, sample_inputs, config)

        # Check success
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if successful:
            print(f"\n✅ Successfully exported {len(successful)}/{len(results)} backends")
            for result in successful:
                print(f"   • {result.backend}: {result.output_path.split('/')[-1]} ({result.file_size_mb:.1f} MB)")

        if failed:
            print(f"\n⚠️  Failed {len(failed)} backend(s):")
            for result in failed:
                print(f"   • {result.backend}: {result.error_message}")

        # Clean up downloaded model files
        if model_path.exists():
            model_path.unlink()
            print(f"   Cleaned up: {model_path.name}")

        return len(successful) > 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_gemma(model_name="gemma-3-270m", output_dir=".."):
    """Export Gemma text generation model using Optimum ExecuTorch.

    Supported models:
    - gemma-3-270m: 240 MB (text-only, 270M parameters)
    - gemma-3-1b: 892 MB - 1.5 GB (text-only, 1B parameters)

    IMPORTANT: Gemma models are gated on HuggingFace and require:
    1. Request access at https://huggingface.co/google/gemma-3-270m-it
    2. Authenticate with: hf auth login
    3. Install optimum-executorch from source:
       git clone https://github.com/huggingface/optimum-executorch.git
       cd optimum-executorch && pip install '.[dev]' && python install_dev.py
    """
    print("\n" + "="*70)
    print(f"  Exporting {model_name.upper()}")
    print("="*70 + "\n")

    try:
        import subprocess
        from pathlib import Path
        import json
        import shutil

        # Map model name to HuggingFace model ID
        model_id = f"google/{model_name}-it"

        print(f"📦 Exporting {model_name} using Optimum ExecuTorch...")
        print(f"   Model ID: {model_id}")
        print(f"   This may take several minutes on first run...")

        # Create temporary output directory (use absolute path for optimum-cli)
        temp_output = Path(output_dir).resolve() / f"{model_name}_temp"
        temp_output.mkdir(parents=True, exist_ok=True)

        # Use optimum-cli to export the model
        # Note: Removed --use_custom_sdpa and --use_custom_kv_cache flags
        # as they require ExecuTorch features not available in v0.7.0
        cmd = [
            "optimum-cli", "export", "executorch",
            "--model", model_id,
            "--task", "text-generation",
            "--recipe", "xnnpack",
            "--qlinear", "8da4w",
            "--qembedding", "8w",
            "--output_dir", str(temp_output)
        ]

        print(f"\n🔄 Running optimum-cli export...")
        print(f"   Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Export failed with exit code {result.returncode}")
            print(f"   stderr: {result.stderr}")
            return False

        print(f"✅ Model exported successfully!")

        # Find and rename the .pte file
        pte_files = list(temp_output.glob("*.pte"))
        if not pte_files:
            print(f"❌ No .pte file found in {temp_output}")
            return False

        pte_file = pte_files[0]
        # Output to gemma subdirectory
        gemma_output_dir = Path(output_dir).resolve() / "gemma"
        gemma_output_dir.mkdir(parents=True, exist_ok=True)
        final_pte = gemma_output_dir / f"{model_name}_xnnpack.pte"

        shutil.move(str(pte_file), str(final_pte))
        file_size_mb = final_pte.stat().st_size / (1024 * 1024)
        print(f"✅ Model file: {final_pte.name} ({file_size_mb:.1f} MB)")

        # Load tokenizer and save vocabulary
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Save simplified vocabulary for Dart
        vocab = tokenizer.get_vocab()
        vocab_file = gemma_output_dir / f"{model_name}_vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(vocab, f, indent=2)

        print(f"✅ Vocabulary saved: {vocab_file.name} ({len(vocab)} tokens)")

        # Save tokenizer config
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token,
            "model_max_length": 2048,
            "special_tokens": {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            }
        }

        tokenizer_config_file = gemma_output_dir / f"{model_name}_tokenizer_config.json"
        with open(tokenizer_config_file, "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        print(f"✅ Tokenizer config saved: {tokenizer_config_file.name}")

        # Clean up temp directory
        shutil.rmtree(temp_output)

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"\n💡 Please install optimum-executorch from source:")
        print(f"   git clone https://github.com/huggingface/optimum-executorch.git")
        print(f"   cd optimum-executorch && pip install '.[dev]' && python install_dev.py")
        return False
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_labels(output_dir=".."):
    """Export COCO and ImageNet labels to model directories."""
    print("\n" + "="*70)
    print("  Generating Label Files")
    print("="*70 + "\n")

    # COCO labels (for YOLO)
    coco_labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    output_path = Path(output_dir)

    # Save COCO labels to yolo directory
    yolo_dir = output_path / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    coco_file = yolo_dir / "labels.txt"
    coco_file.write_text('\n'.join(coco_labels))
    print(f"✅ COCO labels: {coco_file}")
    print(f"   ({len(coco_labels)} classes)")

    # Note: ImageNet labels are large (1000 classes), typically downloaded from torchvision
    # For now, we just note where they should go
    mobilenet_dir = output_path / "mobilenet"
    mobilenet_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 MobileNet labels should be at: {mobilenet_dir / 'labels.txt'}")


def generate_index_json(output_dir=".."):
    """Generate index.json with metadata for all models.

    The index.json contains:
    - name: model filename
    - hash: SHA256 hash for cache invalidation
    - size: file size in bytes
    - category: model category (mobilenet, yolo, gemma)
    - backend: backend type (xnnpack, coreml, mps, vulkan)
    - inputSize: model input size
    - description: human-readable description
    """
    import json
    import hashlib

    print("\n" + "="*70)
    print("  Generating index.json")
    print("="*70 + "\n")

    output_path = Path(output_dir)

    # Model categories and their configurations
    model_configs = {
        "mobilenet": {
            "inputSize": 224,
            "labelsFile": "labels.txt",
            "labelsUrl": "https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/mobilenet/labels.txt",
        },
        "yolo": {
            "inputSize": 640,
            "labelsFile": "labels.txt",
            "labelsUrl": "https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/yolo/labels.txt",
        },
        "gemma": {
            "inputSize": None,
            "labelsFile": None,
            "labelsUrl": None,
        },
        "movenet": {
            "inputSize": 192,  # Default for lightning; thunder uses 256
            "labelsFile": None,
            "labelsUrl": None,
        },
        "blazeface": {
            "inputSize": 128,
            "labelsFile": None,
            "labelsUrl": None,
        },
        "yolo-pose": {
            "inputSize": 640,
            "labelsFile": None,
            "labelsUrl": None,
        },
        "yolo-face": {
            "inputSize": 640,
            "labelsFile": None,
            "labelsUrl": None,
        },
    }

    # Backend descriptions
    backend_info = {
        "xnnpack": {"platforms": ["android", "ios", "macos", "web"], "description": "CPU-optimized (XNNPACK)"},
        "coreml": {"platforms": ["ios", "macos"], "description": "Apple Neural Engine (CoreML)"},
        "mps": {"platforms": ["ios", "macos"], "description": "Apple GPU (Metal Performance Shaders)"},
        "vulkan": {"platforms": ["android", "ios", "macos", "windows", "linux"], "description": "GPU-accelerated (Vulkan)"},
    }

    models = []

    # Scan each category directory
    for category, config in model_configs.items():
        category_dir = output_path / category
        if not category_dir.exists():
            continue

        # Find all .pte files
        for pte_file in sorted(category_dir.glob("*.pte")):
            # Calculate SHA256 hash
            sha256_hash = hashlib.sha256()
            with open(pte_file, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)

            file_hash = sha256_hash.hexdigest()
            file_size = pte_file.stat().st_size
            file_name = pte_file.name

            # Extract backend from filename (e.g., "mobilenet_v3_small_xnnpack.pte" -> "xnnpack")
            backend = file_name.replace(".pte", "").split("_")[-1]

            # Extract model name (everything before the backend)
            model_name = "_".join(file_name.replace(".pte", "").split("_")[:-1])

            # Create model entry
            model_entry = {
                "name": file_name,
                "modelName": model_name,
                "category": category,
                "backend": backend,
                "hash": file_hash,
                "size": file_size,
                "sizeMB": round(file_size / (1024 * 1024), 2),
                "inputSize": config["inputSize"],
                "remoteUrl": f"https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/{category}/{file_name}",
            }

            # Add labels reference for the model
            if config["labelsFile"]:
                model_entry["labelsFile"] = config["labelsFile"]
                model_entry["labelsRemoteUrl"] = config["labelsUrl"]

            # Add backend info if available
            if backend in backend_info:
                model_entry["platforms"] = backend_info[backend]["platforms"]
                model_entry["backendDescription"] = backend_info[backend]["description"]

            models.append(model_entry)
            print(f"  ✓ {file_name} ({model_entry['sizeMB']} MB)")

    # Create labels entries
    labels = []
    for category, config in model_configs.items():
        if config["labelsFile"]:
            labels_path = output_path / category / config["labelsFile"]
            if labels_path.exists():
                # Calculate hash for labels file
                sha256_hash = hashlib.sha256()
                with open(labels_path, "rb") as f:
                    sha256_hash.update(f.read())

                labels.append({
                    "name": config["labelsFile"],
                    "category": category,
                    "hash": sha256_hash.hexdigest(),
                    "size": labels_path.stat().st_size,
                    "remoteUrl": config["labelsUrl"],
                })

    # Create the index structure
    index = {
        "version": "1.0",
        "generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "baseUrl": "https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main",
        "models": models,
        "labels": labels,
        "backends": backend_info,
    }

    # Write index.json
    index_file = output_path / "index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n✅ Generated {index_file}")
    print(f"   Models: {len(models)}")
    print(f"   Labels: {len(labels)}")

    return True


def cmd_export(args):
    """Export command."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        ExecuTorch Flutter - Model Export                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Display backend information
    backends = args.backends if hasattr(args, 'backends') and args.backends else None
    if backends:
        print(f"\n📦 Exporting with backends: {', '.join(backends)}")
    else:
        print(f"\n📦 Exporting with default backends: xnnpack, coreml, mps, vulkan")

    success_count = 0
    total_count = 0

    # Determine what to export
    export_mobilenet_flag = args.all or args.mobilenet
    export_yolo_models = []
    export_yolo_pose_models = []
    export_yolo_face_models = []
    # Note: Gemma, MoveNet, BlazeFace are NOT included in --all because they require
    # additional dependencies. Users must explicitly use their respective flags.
    export_gemma_flag = args.gemma
    export_movenet_variant = getattr(args, 'movenet', None)
    export_blazeface_flag = getattr(args, 'blazeface', False)

    if args.all:
        export_yolo_models = ["yolo11n", "yolov8n", "yolov5n"]  # All nano YOLO models
    if args.yolo:
        export_yolo_models.extend(args.yolo)

    # Handle yolo-pose argument (uses underscore in attribute name due to argparse)
    yolo_pose_arg = getattr(args, 'yolo_pose', None)
    if yolo_pose_arg:
        export_yolo_pose_models.extend(yolo_pose_arg)

    # Handle yolo-face argument
    yolo_face_arg = getattr(args, 'yolo_face', None)
    if yolo_face_arg:
        export_yolo_face_models.extend(yolo_face_arg)

    # Export MobileNet
    if export_mobilenet_flag:
        total_count += 1
        if export_mobilenet(args.output_dir, backends):
            success_count += 1

    # Export YOLO object detection models
    for model_name in export_yolo_models:
        total_count += 1
        if export_yolo(model_name, args.output_dir, backends):
            success_count += 1

    # Export YOLO Pose models
    for model_name in export_yolo_pose_models:
        total_count += 1
        if export_yolo_pose(model_name, args.output_dir, backends):
            success_count += 1

    # Export YOLO Face models
    for model_name in export_yolo_face_models:
        total_count += 1
        if export_yolo_face(model_name, args.output_dir, backends):
            success_count += 1

    # Export MoveNet
    if export_movenet_variant:
        total_count += 1
        if export_movenet(export_movenet_variant, args.output_dir, backends):
            success_count += 1

    # Export BlazeFace
    if export_blazeface_flag:
        total_count += 1
        if export_blazeface(args.output_dir, backends):
            success_count += 1

    # Export Gemma
    if export_gemma_flag:
        total_count += 1
        if export_gemma("gemma-3-270m", args.output_dir):
            success_count += 1

    # Export labels
    if args.all or args.labels:
        export_labels(args.output_dir)

    # Always generate index.json after export
    generate_index_json(args.output_dir)

    # Summary
    print("\n" + "="*70)
    print("  Export Summary")
    print("="*70)
    print(f"\n✅ Successfully exported: {success_count}/{total_count} models")

    if success_count < total_count:
        print(f"❌ Failed exports: {total_count - success_count}")

    print(f"\n📁 Models saved to: {args.output_dir}")
    print(f"\n🚀 Next: cd example && flutter run\n")

    return 0 if success_count == total_count else 1


def cmd_validate(args):
    """Validate command."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        ExecuTorch Flutter - Model Validation                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Create validator
    validator = ModelValidator(
        models_dir=args.models_dir,
        images_dir=args.images_dir,
        assets_dir="../example/assets"
    )

    # Run validation
    results = validator.validate_all()

    if not results:
        print("\n❌ Validation failed - no results generated")
        return 1

    # Save results
    output_path = Path(args.output_file)
    output_path.write_text(__import__('json').dumps(results, indent=2))

    # Print summary
    print(f"\n{'=' * 70}")
    print("  Validation Summary")
    print(f"{'=' * 70}\n")

    summary = results['summary']
    print(f"✅ Total Models: {summary['total_models_tested']}")
    print(f"   - Classification: {summary['classification_models_count']}")
    print(f"   - Detection: {summary['detection_models_count']}")
    print(f"\n✅ Successful: {summary['successful_models']}")
    print(f"❌ Failed: {summary['failed_models']}")
    print(f"\n📸 Test Images: {summary['total_test_images']}")
    print(f"\n📄 Results: {output_path}\n")

    return 0 if summary['failed_models'] == 0 else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ExecuTorch Flutter - Model Export & Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Export all models with all backends (default)
  python main.py export --mobilenet                 # Export MobileNet with all backends
  python main.py export --mobilenet --backends xnnpack coreml  # MobileNet with specific backends
  python main.py export --yolo yolo11n              # Export YOLO11n with all backends
  python main.py export --all --backends xnnpack    # Export all models with XNNPACK only
  python main.py export --gemma                     # Export Gemma text generation model
  python main.py export --movenet lightning         # Export MoveNet Lightning (pose estimation)
  python main.py export --blazeface                 # Export BlazeFace (face detection)
  python main.py export --yolo-pose yolo11n-pose    # Export YOLO Pose model
  python main.py export --yolo-face yolov8n-face    # Export YOLO Face model
  python main.py validate                           # Validate all models

Supported YOLO Object Detection models:
  yolo11n, yolo11s, yolov8n, yolov8s, yolov5n, yolov5s

Supported YOLO Pose models:
  yolo11n-pose, yolo11s-pose, yolov8n-pose, yolov8s-pose

Supported YOLO Face models:
  yolov8n-face, yolov8s-face

Supported MoveNet models:
  lightning (192x192, faster), thunder (256x256, more accurate)

Supported Text Generation models:
  gemma-3-270m (Google Gemma 3, 270M parameters, text-only, ~240 MB)
  gemma-3-1b   (Google Gemma 3, 1B parameters, text-only, ~1.5 GB)

Backend Information:
  xnnpack  - CPU-optimized, works on ALL platforms (Android, iOS, macOS, Web)
  coreml   - Apple Neural Engine optimization (iOS, macOS only)
  mps      - Metal Performance Shaders GPU acceleration (iOS, macOS only)
  vulkan   - Cross-platform GPU acceleration (Android, iOS, macOS, Windows, Linux)
  qnn      - Qualcomm AI Engine (Snapdragon devices)
  arm      - ARM Ethos-U NPU (embedded devices)

Default backends: xnnpack, coreml, mps, vulkan
Note: Only available backends on your system will be used.

Performance Guidelines:
  - iOS/macOS: Use CoreML for best performance (Apple Neural Engine)
  - iOS/macOS GPU: Use MPS for GPU-accelerated inference
  - Android GPU: Use Vulkan for GPU-accelerated inference
  - Qualcomm devices: Use QNN for NPU acceleration
  - All platforms (including Web): XNNPACK works everywhere with good performance

Note: MoveNet requires additional packages:
  pip install tensorflow tensorflow_hub tf2onnx onnx onnx2torch

Note: BlazeFace requires additional packages:
  pip install onnx onnx2torch

Note: Gemma models require additional setup (advanced):
  1. Install optimum-executorch from source:
     git clone https://github.com/huggingface/optimum-executorch.git
     cd optimum-executorch && pip install '.[dev]' && python install_dev.py
  2. Request access: https://huggingface.co/google/gemma-3-270m-it
  3. Login: hf auth login
  4. Export: python main.py export --gemma
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export models')
    export_parser.add_argument('--all', action='store_true', help='Export all models')
    export_parser.add_argument('--mobilenet', action='store_true', help='Export MobileNet')
    export_parser.add_argument('--yolo', nargs='+', metavar='MODEL', help='Export YOLO model(s)')
    export_parser.add_argument('--gemma', action='store_true', help='Export Gemma text generation model')
    export_parser.add_argument('--movenet', nargs='?', const='lightning', metavar='VARIANT',
                                help='Export MoveNet pose estimation (lightning or thunder, default: lightning)')
    export_parser.add_argument('--blazeface', action='store_true', help='Export BlazeFace face detection')
    export_parser.add_argument('--yolo-pose', nargs='+', metavar='MODEL',
                                help='Export YOLO Pose model(s) (e.g., yolo11n-pose, yolov8n-pose)')
    export_parser.add_argument('--yolo-face', nargs='+', metavar='MODEL',
                                help='Export YOLO Face model(s) (e.g., yolov8n-face, yolov8s-face)')
    export_parser.add_argument('--labels', action='store_true', help='Generate label files')
    export_parser.add_argument('--backends', nargs='+',
                                choices=['xnnpack', 'coreml', 'mps', 'vulkan', 'qnn', 'arm'],
                                help='Backend(s) to export for (default: xnnpack, coreml, mps, vulkan)')
    export_parser.add_argument('--output-dir', default='..',
                                help='Output directory (default: .. - parent models/ directory)')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate models')
    validate_parser.add_argument('--models-dir', default='../assets/models',
                                  help='Models directory (default: ../assets/models)')
    validate_parser.add_argument('--images-dir', default='../assets/images',
                                  help='Test images directory (default: ../assets/images)')
    validate_parser.add_argument('--output-file', default='../assets/model_test_results.json',
                                  help='Output file (default: ../assets/model_test_results.json)')

    args = parser.parse_args()

    # Default to export if no command specified
    if args.command is None:
        args.command = 'export'
        args.all = True
        args.mobilenet = False
        args.yolo = None
        args.gemma = False
        args.movenet = None
        args.blazeface = False
        args.yolo_pose = None
        args.yolo_face = None
        args.labels = True
        args.output_dir = '..'
        args.backends = None

    # Run command
    if args.command == 'export':
        return cmd_export(args)
    elif args.command == 'validate':
        return cmd_validate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
