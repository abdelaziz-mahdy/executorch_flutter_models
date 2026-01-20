# ExecuTorch Flutter Models - AI Agent Context

## Overview

This repository contains pre-exported ExecuTorch models (.pte files) and Python export tools for use with the `executorch_flutter` package. It provides ready-to-use ML models optimized for on-device inference.

**Repository**: `abdelaziz-mahdy/executorch_flutter_models`
**Parent Project**: [executorch_flutter](https://github.com/abdelaziz-mahdy/executorch_flutter)
**License**: Models follow their original licenses (Apache 2.0 for MobileNet, AGPL-3.0 for YOLO)

## Project Structure

```
models/
├── index.json              # Model metadata (names, hashes, sizes, URLs)
├── python/                 # Model Export Tools
│   ├── main.py             # Main CLI tool
│   ├── executorch_exporter.py  # Core exporter class
│   ├── validate_all_models.py  # Model validation
│   ├── requirements.txt    # Python dependencies
│   ├── install_executorch.sh   # ExecuTorch installation
│   ├── README.md           # Export documentation
│   ├── BACKENDS.md         # Backend information
│   └── EXPORT_SUMMARY.md   # Export summary docs
├── mobilenet/              # Image Classification Models
│   ├── labels.txt          # ImageNet 1000 classes
│   └── mobilenet_v3_small_*.pte
└── yolo/                   # Object Detection Models
    ├── labels.txt          # COCO 80 classes
    └── yolo*_*.pte         # YOLO variants
```

## Key Files

### `index.json`

Central metadata file for all models. Used by the Flutter app for:
- **Dynamic model discovery**: List available models
- **Cache validation**: SHA256 hashes for cache invalidation
- **Download URLs**: Remote URLs for model downloads

Structure:
```json
{
  "models": [
    {
      "name": "mobilenet_v3_small_xnnpack.pte",
      "hash": "sha256...",
      "size": 10485760,
      "sizeMB": 10.0,
      "category": "mobilenet",
      "backend": "xnnpack",
      "inputSize": 224,
      "remoteUrl": "https://...",
      "platforms": ["android", "ios", "macos", "web", "linux", "windows"]
    }
  ]
}
```

### `python/main.py`

Main CLI tool for exporting models. Key commands:

```bash
# Export all models
python main.py export --all

# Export specific model type
python main.py export --mobilenet
python main.py export --yolo yolo11n

# Export with specific backends
python main.py export --mobilenet --backends xnnpack coreml

# Validate exported models
python main.py validate

# Update index.json
python main.py update-index
```

### `python/executorch_exporter.py`

Core exporter class that handles:
- PyTorch model loading (torchvision, ultralytics)
- ExecuTorch export with backend delegation
- Model validation and verification

## Available Models

### Image Classification (mobilenet/)

| Model | Backend | Platforms |
|-------|---------|-----------|
| `mobilenet_v3_small_xnnpack.pte` | XNNPACK | All (including web) |
| `mobilenet_v3_small_coreml.pte` | CoreML | iOS, macOS |
| `mobilenet_v3_small_mps.pte` | MPS | macOS |
| `mobilenet_v3_small_vulkan.pte` | Vulkan | Android, iOS, macOS, Windows, Linux |

### Object Detection (yolo/)

Available variants: `yolo11n`, `yolov8n`, `yolov5n`
Each available with backends: `xnnpack`, `coreml`, `mps`, `vulkan`

## Backend Selection Guide

| Backend | Best For | Platforms |
|---------|----------|-----------|
| **XNNPACK** | Universal compatibility | All platforms |
| **CoreML** | Apple Neural Engine | iOS, macOS |
| **MPS** | Apple GPU | macOS, iOS |
| **Vulkan** | Cross-platform GPU | Android, iOS, macOS, Windows, Linux |
| **QNN** | Qualcomm NPU | Android (Snapdragon) |

## Adding a New Model

### 1. Create Export Function

In `python/main.py`, add an export function:

```python
def export_my_model(backends: List[str], output_dir: str):
    """Export MyModel to ExecuTorch format."""
    import torch
    from my_model import MyModel

    model = MyModel()
    model.eval()

    # Create example input
    example_input = torch.randn(1, 3, 224, 224)

    # Export for each backend
    for backend in backends:
        exporter = ExecutorchExporter(
            model=model,
            example_inputs=(example_input,),
            backend=backend,
        )
        output_path = f"{output_dir}/my_model_{backend}.pte"
        exporter.export(output_path)
```

### 2. Add CLI Support

Add to the argument parser in `main.py`:

```python
parser.add_argument('--my-model', action='store_true', help='Export MyModel')
```

### 3. Create Model Directory

```bash
mkdir my_model
# Add labels.txt if needed
```

### 4. Update index.json

Run `python main.py update-index` or manually add entries.

### 5. Test the Model

```bash
python main.py validate --model my_model
```

## Workflow: Updating Models

1. **Modify export code** in `python/main.py` or `python/executorch_exporter.py`
2. **Re-export models**:
   ```bash
   cd python
   python main.py export --all
   ```
3. **Validate**:
   ```bash
   python main.py validate
   ```
4. **Update index.json**:
   ```bash
   python main.py update-index
   ```
5. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: Update models"
   git push
   ```
6. **Update submodule in parent repo**:
   ```bash
   cd ../..  # executorch_flutter root
   git add models
   git commit -m "chore: Update models submodule"
   ```

## Model Export Requirements

### Python Environment

```bash
cd python
pip install -r requirements.txt
./install_executorch.sh  # Install ExecuTorch
```

### Required Packages

- `torch` - PyTorch
- `torchvision` - For MobileNet models
- `ultralytics` - For YOLO models
- `executorch` - ExecuTorch export library

## Integration with executorch_flutter

The Flutter app uses these models by:

1. **Downloading from GitHub** using URLs in `index.json`
2. **Caching locally** with hash-based validation
3. **Loading via ExecuTorchModel.load()** from bytes

Example flow:
```dart
// 1. Fetch index.json
final index = await fetchModelIndex();

// 2. Find model for current platform/backend
final model = index.findModel(category: 'mobilenet', backend: 'xnnpack');

// 3. Download and cache
final bytes = await downloadWithCache(model.remoteUrl, model.hash);

// 4. Load model
final etModel = await ExecuTorchModel.load(bytes);
```

## Troubleshooting

### Export Fails

**Missing backend**:
```bash
# Check ExecuTorch installation
python -c "import executorch; print(executorch.__version__)"

# Reinstall with specific backends
./install_executorch.sh --backends xnnpack coreml
```

**YOLO export issues**:
```bash
# Update ultralytics
pip install --upgrade ultralytics
```

### Validation Fails

**Shape mismatch**:
- Check input size in export code matches model requirements
- Verify preprocessing matches training pipeline

**Backend not available**:
- Some backends require specific hardware (CoreML needs Apple device)
- Use XNNPACK for universal compatibility

---

**Last Updated**: 2026-01-17
**Models Version**: See `index.json` for individual model versions
