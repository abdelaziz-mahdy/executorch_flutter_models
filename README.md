# ExecuTorch Flutter Models

Pre-exported ExecuTorch models (.pte files) and export tools for the `executorch_flutter` package.

## Version-Based Structure

Models are organized by ExecuTorch version to ensure compatibility:

```
models/
├── versions.json           # Available versions and latest
├── 1.1.0/                  # ExecuTorch 1.1.0 models
│   ├── index.json          # Model metadata for this version
│   ├── mobilenet/          # Image Classification
│   │   ├── labels.txt
│   │   └── mobilenet_v3_small_*.pte
│   ├── yolo/               # Object Detection
│   │   ├── labels.txt
│   │   └── yolo*_*.pte
│   ├── yolo-pose/          # Pose Estimation
│   │   └── yolo*-pose_*.pte
│   └── yolo-face/          # Face Detection
│       └── yolo*-face_*.pte
├── python/                 # Export Tools
│   ├── main.py             # CLI tool
│   ├── requirements.txt
│   └── ...
└── LICENSES/               # Model licenses
```

## Quick Start

### Get Latest Version

```dart
// Fetch versions.json to get latest version
final versionsUrl = 'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/versions.json';
final response = await http.get(Uri.parse(versionsUrl));
final versions = jsonDecode(response.body);
final latest = versions['latest'];  // e.g., "1.1.0"
```

### Fetch Model Index

```dart
// Use the version to get the correct index.json
final indexUrl = 'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/$latest/index.json';
final indexResponse = await http.get(Uri.parse(indexUrl));
final index = jsonDecode(indexResponse.body);

// Get models for your platform/backend
final models = index['models'] as List;
final xnnpackModels = models.where((m) => m['backend'] == 'xnnpack').toList();
```

## Available Models

### Image Classification (mobilenet/)

| Model | Backend | Platforms |
|-------|---------|-----------|
| `mobilenet_v3_small_xnnpack.pte` | XNNPACK | All (including web) |
| `mobilenet_v3_small_coreml.pte` | CoreML | iOS, macOS |
| `mobilenet_v3_small_mps.pte` | MPS | macOS |
| `mobilenet_v3_small_vulkan.pte` | Vulkan | Android, iOS, macOS, Windows, Linux |

### Object Detection (yolo/)

| Model | Backends | Description |
|-------|----------|-------------|
| `yolo11n_*.pte` | xnnpack, coreml, mps, vulkan | YOLO11 Nano |
| `yolov8n_*.pte` | xnnpack, coreml, mps, vulkan | YOLOv8 Nano |
| `yolov5n_*.pte` | xnnpack, coreml, mps, vulkan | YOLOv5 Nano |

### Pose Estimation (yolo-pose/)

| Model | Backends | Description |
|-------|----------|-------------|
| `yolo11n-pose_*.pte` | xnnpack, coreml, mps, vulkan | YOLO11 Pose |
| `yolov8n-pose_*.pte` | xnnpack, coreml, mps, vulkan | YOLOv8 Pose |

### Face Detection (yolo-face/)

| Model | Backends | Description |
|-------|----------|-------------|
| `yolov11n-face_*.pte` | xnnpack, coreml, mps, vulkan | YOLO11 Face |
| `yolov10n-face_*.pte` | xnnpack, coreml, mps, vulkan | YOLOv10 Face |

## Backend Selection

| Backend | Best For | Platforms |
|---------|----------|-----------|
| **XNNPACK** | Universal compatibility | All platforms |
| **CoreML** | Apple Neural Engine | iOS, macOS |
| **MPS** | Apple GPU | macOS |
| **Vulkan** | Cross-platform GPU | Android, iOS, macOS, Windows, Linux |

### Recommendations

- **iOS/macOS**: CoreML for best performance (Apple Neural Engine)
- **Android GPU**: Vulkan for GPU acceleration
- **Web/Universal**: XNNPACK works everywhere
- **macOS GPU**: MPS for Metal acceleration

## Export Models (CI/CD)

Models are automatically exported via GitHub Actions when triggered:

```bash
# Trigger via GitHub CLI
gh workflow run "Export Models" \
  -f executorch_version=1.1.0 \
  -f models=all \
  -f backends=all
```

### Manual Export

```bash
cd python
pip install -r requirements.txt
pip install executorch==1.1.0

# Export all models
python main.py export --all --output-dir ../1.1.0

# Export specific model
python main.py export --mobilenet --backends xnnpack coreml
python main.py export --yolo yolo11n --backends xnnpack
python main.py export --yolo-pose yolo11n-pose --backends xnnpack
python main.py export --yolo-face yolov11n-face --backends xnnpack
```

## Version Compatibility

**Important**: Models must match the ExecuTorch runtime version.

| Runtime Version | Models Directory |
|-----------------|------------------|
| ExecuTorch 1.1.0 | `1.1.0/` |

Using models from a different version will cause loading failures.

## Model Index Format

Each version's `index.json` contains:

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

## License

Models are provided under their original licenses:
- **MobileNet**: Apache 2.0
- **YOLO**: AGPL-3.0

See [LICENSES/](LICENSES/) for details.

---

**Repository**: https://github.com/abdelaziz-mahdy/executorch_flutter_models
