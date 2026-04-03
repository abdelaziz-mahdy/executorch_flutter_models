# ExecuTorch Flutter - Model Export Tools

Unified command-line tool for model export and validation with **multi-backend support**.

## Quick Start

```bash
# Export all models with all available backends
python main.py

# Export specific model with all backends
python main.py export --mobilenet

# Export with specific backends only
python main.py export --mobilenet --backends xnnpack coreml

# Export all models with XNNPACK only
python main.py export --all --backends xnnpack

# Validate all models
python main.py validate
```

## Backend Support

This tool exports models for multiple ExecuTorch backends:

| Backend | Platforms | Description |
|---------|-----------|-------------|
| **XNNPACK** | Android, iOS, macOS, Web | CPU-optimized, works everywhere |
| **CoreML** | iOS, macOS | Apple Neural Engine optimization |
| **MPS** | iOS, macOS | Metal GPU acceleration |
| **Vulkan** | Android, Linux | Cross-platform GPU acceleration |

**Default backends**: xnnpack, coreml, mps, vulkan

📚 **See [BACKENDS.md](BACKENDS.md) for complete backend selection guide**

## Web Support

XNNPACK backend now works on web with WASM SIMD support. No special configuration needed - the same XNNPACK models work on both native and web platforms.

## Installation

```bash
pip install torch torchvision executorch ultralytics opencv-python torchao
```

Or use the install script:

```bash
./install_executorch.sh
```

## Commands

### Export (Default)

Export models to ExecuTorch format.

```bash
# Export all models (MobileNet + all YOLO variants)
python main.py
python main.py export --all

# Export MobileNet only
python main.py export --mobilenet

# Export YOLO only
python main.py export --yolo yolo11n
python main.py export --yolo yolo11n yolov8n yolov5n  # Multiple models

# Export labels only
python main.py export --labels

# Export with specific backends
python main.py export --all --backends xnnpack coreml
```

**Supported YOLO models**: yolo11n, yolov8n, yolov5n (nano versions)

**Output directories**:
- `../mobilenet/` - MobileNet models
- `../yolo/` - YOLO models
- `../index.json` - Auto-generated model metadata

### Validate

Validate exported models with test images.

```bash
# Validate all models
python main.py validate

# Custom directories
python main.py validate --models-dir ../mobilenet \
                        --images-dir ../assets/images
```

## Output Structure

After running `python main.py`, you'll have:

```
models/
├── index.json                      # Auto-generated metadata
├── mobilenet/
│   ├── labels.txt                  # ImageNet 1000 classes
│   ├── mobilenet_v3_small_xnnpack.pte
│   ├── mobilenet_v3_small_coreml.pte
│   ├── mobilenet_v3_small_mps.pte
│   └── mobilenet_v3_small_vulkan.pte
└── yolo/
    ├── labels.txt                  # COCO 80 classes
    ├── yolo11n_xnnpack.pte
    ├── yolo11n_coreml.pte
    ├── yolo11n_mps.pte
    ├── yolo11n_vulkan.pte
    ├── yolov8n_*.pte               # All backends
    └── yolov5n_*.pte               # All backends
```

## index.json

The exporter automatically generates `index.json` with metadata for all models:

```json
{
  "version": "1.0",
  "generated": "2024-01-10T00:00:00Z",
  "models": [
    {
      "name": "mobilenet_v3_small_xnnpack.pte",
      "modelName": "mobilenet_v3_small",
      "category": "mobilenet",
      "backend": "xnnpack",
      "hash": "sha256...",
      "size": 10205728,
      "sizeMB": 9.73,
      "inputSize": 224,
      "remoteUrl": "https://...",
      "platforms": ["android", "ios", "macos", "web"]
    }
  ],
  "labels": [...],
  "backends": {...}
}
```

This enables:
- Dynamic model discovery in the app
- Hash-based cache invalidation
- Accurate file size information

## Model Specifications

### MobileNet V3 Small
- **Input**: [1, 3, 224, 224] (RGB, ImageNet normalized)
- **Output**: [1, 1000] logits (requires softmax)
- **Preprocessing**: Resize(256) → CenterCrop(224) → Normalize
- **File size**: ~5-10 MB per backend

### YOLO Nano Models
- **Input**: [1, 3, 640, 640] (RGB, normalized to [0,1])
- **Output**: [1, 84, 8400] (4 bbox coords + 80 COCO classes)
- **Preprocessing**: Letterbox resize to 640x640
- **File size**: ~5-13 MB per backend
- **Note**: Requires post-processing (DFL, sigmoid, NMS) in Flutter app

## File Structure

```
python/
├── main.py                      # Main CLI tool
├── executorch_exporter.py       # Core exporter framework
├── validate_all_models.py       # Model validation
├── requirements.txt             # Python dependencies
├── install_executorch.sh        # ExecuTorch installation
├── README.md                    # This file
└── BACKENDS.md                  # Backend selection guide
```

## Examples

### Full Export Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Export all models
python main.py

# 3. Verify files and index.json
ls -lh ../mobilenet/ ../yolo/ ../index.json

# 4. Run Flutter app
cd ../../example
flutter run
```

### Export for Specific Platform

```bash
# iOS/macOS only (CoreML for best performance)
python main.py export --all --backends coreml

# Android only (XNNPACK + Vulkan)
python main.py export --all --backends xnnpack vulkan

# Web only (XNNPACK with WASM SIMD)
python main.py export --all --backends xnnpack
```

## Troubleshooting

### Missing dependencies
```bash
pip install -r requirements.txt
# Or manually:
pip install torch torchvision executorch ultralytics opencv-python
```

### Export fails
1. Check Python version (3.10+ recommended)
2. Verify ExecuTorch installation: `python -c "import executorch"`
3. Try exporting a single model first: `python main.py export --mobilenet`

### Models don't load in Flutter
1. Verify .pte files exist in model directories
2. Check file sizes match index.json
3. Re-export with `python main.py export --all`
4. Check index.json was generated

## CI/CD Integration

```bash
# In your CI pipeline
cd models/python
python main.py export --all

# Verify index.json was generated
if [ -f "../index.json" ]; then
  echo "✅ Models exported successfully"
else
  echo "❌ Export failed"
  exit 1
fi
```

## Support

For issues:
- Check Flutter app logs: `flutter logs`
- Re-export models: `python main.py export --all`
- Verify index.json: `cat ../index.json`
- Review BACKENDS.md for platform-specific guidance
