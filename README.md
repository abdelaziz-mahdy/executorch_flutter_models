# ExecuTorch Flutter Models

This repository contains pre-exported ExecuTorch models (.pte files) and export tools for use with the `executorch_flutter` package.

## Directory Structure

```
models/
├── index.json              # Model metadata (names, hashes, sizes)
├── python/                 # Model Export Tools
│   ├── main.py             # Main CLI tool
│   ├── executorch_exporter.py  # Core exporter class
│   ├── validate_all_models.py  # Model validation
│   ├── requirements.txt    # Python dependencies
│   ├── install_executorch.sh   # ExecuTorch installation
│   ├── README.md           # Export documentation
│   ├── BACKENDS.md         # Backend information
│   └── EXPORT_SUMMARY.md   # Export summary docs
│
├── mobilenet/              # Image Classification
│   ├── labels.txt          # ImageNet 1000 classes
│   ├── mobilenet_v3_small_xnnpack.pte
│   ├── mobilenet_v3_small_coreml.pte
│   ├── mobilenet_v3_small_mps.pte
│   └── mobilenet_v3_small_vulkan.pte
│
└── yolo/                   # Object Detection
    ├── labels.txt          # COCO 80 classes
    ├── yolo11n_*.pte       # YOLO11 Nano variants
    ├── yolov8n_*.pte       # YOLOv8 Nano variants
    └── yolov5n_*.pte       # YOLOv5 Nano variants
```

## Model Index (index.json)

The `index.json` file contains metadata for all models:
- **name**: Model filename
- **hash**: SHA256 hash for cache invalidation
- **size**: File size in bytes
- **sizeMB**: File size in megabytes
- **category**: Model category (mobilenet, yolo, gemma)
- **backend**: Backend type (xnnpack, coreml, mps, vulkan)
- **inputSize**: Model input size
- **remoteUrl**: Download URL
- **platforms**: Supported platforms

This enables dynamic model discovery and efficient cache management.

## Quick Start: Export Models

### Prerequisites

```bash
cd python
pip install -r requirements.txt
./install_executorch.sh  # Install ExecuTorch
```

### Export All Models

```bash
cd python
python main.py                    # Export all models with all backends
python main.py export --all       # Same as above
```

### Export Specific Models

```bash
# Export MobileNet only
python main.py export --mobilenet

# Export YOLO11 Nano
python main.py export --yolo yolo11n

# Export with specific backends
python main.py export --mobilenet --backends xnnpack coreml
```

### Validate Models

```bash
python main.py validate
```

## Available Models

### Image Classification (mobilenet/)

| Model | Backend | Size | Description |
|-------|---------|------|-------------|
| `mobilenet_v3_small_xnnpack.pte` | XNNPACK | ~10 MB | CPU-optimized (all platforms including web) |
| `mobilenet_v3_small_coreml.pte` | CoreML | ~5 MB | Apple Neural Engine |
| `mobilenet_v3_small_mps.pte` | MPS | ~10 MB | Apple GPU |
| `mobilenet_v3_small_vulkan.pte` | Vulkan | ~10 MB | Android/Linux GPU |

### Object Detection (yolo/)

| Model | Backend | Size | Description |
|-------|---------|------|-------------|
| `yolo11n_*.pte` | Various | ~5-11 MB | YOLO11 Nano variants |
| `yolov8n_*.pte` | Various | ~6-13 MB | YOLOv8 Nano variants |
| `yolov5n_*.pte` | Various | ~5-11 MB | YOLOv5 Nano variants |

Each YOLO model is available in these backends: `xnnpack`, `coreml`, `mps`, `vulkan`

## Backend Information

| Backend | Platforms | Description |
|---------|-----------|-------------|
| **XNNPACK** | Android, iOS, macOS, Web | CPU-optimized, works everywhere |
| **CoreML** | iOS, macOS | Apple Neural Engine optimization |
| **MPS** | iOS, macOS | Metal Performance Shaders (GPU) |
| **Vulkan** | Android, Linux | Cross-platform GPU acceleration |
| **QNN** | Android (Snapdragon) | Qualcomm AI Engine |

### Performance Guidelines

- **iOS/macOS**: Use CoreML for best performance (Apple Neural Engine)
- **iOS/macOS GPU**: Use MPS for GPU-accelerated inference
- **Android GPU**: Use Vulkan for GPU-accelerated inference
- **Qualcomm devices**: Use QNN for NPU acceleration
- **All platforms (including Web)**: XNNPACK works everywhere with good performance

## Usage in Flutter

### Using index.json for Dynamic Model Discovery

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

// Fetch model index
final indexUrl = 'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/index.json';
final response = await http.get(Uri.parse(indexUrl));
final index = jsonDecode(response.body);

// Get all models
final models = index['models'] as List;

// Filter by category
final yoloModels = models.where((m) => m['category'] == 'yolo').toList();

// Filter by platform
final webModels = models.where((m) =>
    (m['platforms'] as List).contains('web')).toList();

// Get model URL and hash for cache validation
final model = models.first;
final url = model['remoteUrl'];
final hash = model['hash'];
final size = model['size'];
```

### Loading Labels

```dart
// For MobileNet
final labelsUrl = 'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/mobilenet/labels.txt';

// For YOLO
final labelsUrl = 'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/yolo/labels.txt';
```

## Export Documentation

For detailed export documentation, see:

- [python/README.md](python/README.md) - Full export guide
- [python/BACKENDS.md](python/BACKENDS.md) - Backend details and platform support
- [python/EXPORT_SUMMARY.md](python/EXPORT_SUMMARY.md) - Export summary and results

## License

Models are provided under the same license as their original PyTorch implementations:
- MobileNet: Apache 2.0
- YOLO: AGPL-3.0

---

**Repository**: https://github.com/abdelaziz-mahdy/executorch_flutter_models
