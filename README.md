# ExecuTorch Flutter Models

This repository contains pre-exported ExecuTorch models (.pte files) for use with the `executorch_flutter` package.

## Directory Structure

```
models/
├── mobilenet/              # Image Classification
│   ├── labels.txt          # ImageNet 1000 classes
│   ├── mobilenet_v3_small_portable.pte
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

## Available Models

### Image Classification (mobilenet/)

| Model | Backend | Size | Description |
|-------|---------|------|-------------|
| `mobilenet_v3_small_portable.pte` | Portable | ~10 MB | Web/Wasm compatible |
| `mobilenet_v3_small_xnnpack.pte` | XNNPACK | ~10 MB | CPU-optimized |
| `mobilenet_v3_small_coreml.pte` | CoreML | ~5 MB | Apple NPU |
| `mobilenet_v3_small_mps.pte` | MPS | ~10 MB | Apple GPU |
| `mobilenet_v3_small_vulkan.pte` | Vulkan | ~10 MB | Android/Linux GPU |

### Object Detection (yolo/)

| Model | Backend | Size | Description |
|-------|---------|------|-------------|
| `yolo11n_*.pte` | Various | ~5-11 MB | YOLO11 Nano variants |
| `yolov8n_*.pte` | Various | ~6-13 MB | YOLOv8 Nano variants |
| `yolov5n_*.pte` | Various | ~5-11 MB | YOLOv5 Nano variants |

Each YOLO model is available in these backends: `portable`, `xnnpack`, `coreml`, `mps`, `vulkan`

## Usage

Models and labels are automatically downloaded at runtime by the `executorch_flutter_example` app from:

```
https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/mobilenet/
https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/yolo/
```

### Loading Labels

```dart
// For MobileNet
final labelsUrl = 'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/mobilenet/labels.txt';

// For YOLO
final labelsUrl = 'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/yolo/labels.txt';
```

## Backend Information

- **Portable**: Generic CPU backend, works on web/wasm (uses portable kernels in WASM runtime)
- **XNNPACK**: CPU-optimized, works on native platforms (Android, iOS, macOS)
- **CoreML**: Apple Neural Engine optimization (iOS, macOS only)
- **MPS**: Metal Performance Shaders for GPU acceleration (iOS, macOS only)
- **Vulkan**: Cross-platform GPU acceleration (Android, Linux)

## Building Models

Models are built using the export scripts in the main repository:

```bash
cd executorch_flutter
./scripts/build_all_models.sh
```

To build only specific backends:
```bash
./scripts/build_all_models.sh --backends "xnnpack coreml"
```

## License

Models are provided under the same license as their original PyTorch implementations:
- MobileNet: Apache 2.0
- YOLO: AGPL-3.0

---

**Repository**: https://github.com/abdelaziz-mahdy/executorch_flutter_models
