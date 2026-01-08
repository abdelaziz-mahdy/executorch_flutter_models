# ExecuTorch Flutter Models

This repository contains pre-exported ExecuTorch models (.pte files) for use with the `executorch_flutter` package.

## Available Models

### Image Classification (MobileNet)
| Model | Backend | Size | Description |
|-------|---------|------|-------------|
| `mobilenet_v3_small_portable.pte` | Portable | ~9.5 MB | Web/Wasm compatible |
| `mobilenet_v3_small_xnnpack.pte` | XNNPACK | ~9.5 MB | CPU-optimized |
| `mobilenet_v3_small_coreml.pte` | CoreML | ~10.2 MB | Apple NPU |
| `mobilenet_v3_small_mps.pte` | MPS | ~9.8 MB | Apple GPU |
| `mobilenet_v3_small_vulkan.pte` | Vulkan | ~9.6 MB | Android/Linux GPU |

### Object Detection (YOLO)
| Model | Backend | Size | Description |
|-------|---------|------|-------------|
| `yolo11n_portable.pte` | Portable | ~5.4 MB | Web/Wasm compatible |
| `yolo11n_xnnpack.pte` | XNNPACK | ~5.4 MB | CPU-optimized |
| `yolo11n_coreml.pte` | CoreML | ~5.8 MB | Apple NPU |
| `yolo11n_mps.pte` | MPS | ~5.6 MB | Apple GPU |
| `yolo11n_vulkan.pte` | Vulkan | ~5.5 MB | Android/Linux GPU |
| `yolov8n_*.pte` | Various | ~6.2 MB | YOLOv8 Nano variants |
| `yolov5n_*.pte` | Various | ~3.9 MB | YOLOv5 Nano variants |

### Text Generation (Gemma)
| Model | Backend | Size | Description |
|-------|---------|------|-------------|
| `gemma-3-270m_xnnpack.pte` | XNNPACK | ~540 MB | Gemma 3 270M params |

## Label Files

- `imagenet_classes.txt` - 1000 ImageNet class labels for MobileNet
- `coco_labels.txt` - 80 COCO class labels for YOLO

## Vocabulary Files

- `gemma-3-270m_vocab.json` - Tokenizer vocabulary for Gemma model

## Usage

Models are automatically downloaded at runtime by the `executorch_flutter_example` app from:
```
https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/
```

## Backend Information

- **Portable**: Generic CPU backend, works on web/wasm
- **XNNPACK**: CPU-optimized, works on native platforms (Android, iOS, macOS)
- **CoreML**: Apple Neural Engine optimization (iOS, macOS)
- **MPS**: Metal Performance Shaders for GPU acceleration (iOS, macOS)
- **Vulkan**: Cross-platform GPU acceleration (Android, Linux)

## Building Models

Models are built using the export scripts in the main repository:

```bash
cd executorch_flutter
./scripts/build_all_models.sh
```

## License

Models are provided under the same license as their original PyTorch implementations:
- MobileNet: Apache 2.0
- YOLO: AGPL-3.0
- Gemma: Apache 2.0

---

**Repository**: https://github.com/abdelaziz-mahdy/executorch_flutter_models
