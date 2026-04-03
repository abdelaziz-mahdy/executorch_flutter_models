# ExecuTorch Flutter Models

Pre-exported ExecuTorch models and export tools for the `executorch_flutter` package.

## Architecture

Model `.pte` files and labels are stored as **GitHub Release assets** (not in git).
Only metadata (`index.json`, `versions.json`) and export scripts are in the repository.

```
Repository (git):
├── versions.json           # Available versions and latest
├── {version}/index.json    # Model metadata (URLs, hashes, sizes)
├── python/                 # Export tools
└── LICENSES/               # Model licenses

GitHub Releases:
├── v1.0.1/                 # Release assets for ExecuTorch 1.0.1
│   ├── mobilenet_v3_small_xnnpack.pte
│   ├── yolo11n_xnnpack.pte
│   ├── mobilenet-labels.txt
│   ├── yolo-labels.txt
│   └── ...
├── v1.1.0/                 # Release assets for ExecuTorch 1.1.0
└── v1.2.0/                 # Release assets for ExecuTorch 1.2.0
```

## Quick Start

### Get Latest Version

```dart
// Fetch versions.json from the repo
final versionsUrl = 'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/versions.json';
final response = await http.get(Uri.parse(versionsUrl));
final versions = jsonDecode(response.body);
final latest = versions['latest'];  // e.g., "1.2.0"
```

### Fetch Model Index

```dart
// Use the version to get index.json (metadata with download URLs)
final indexUrl = 'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/$latest/index.json';
final indexResponse = await http.get(Uri.parse(indexUrl));
final index = jsonDecode(indexResponse.body);

// Each model entry has a remoteUrl pointing to the GitHub Release asset
final models = index['models'] as List;
final xnnpackModels = models.where((m) => m['backend'] == 'xnnpack').toList();
// remoteUrl: https://github.com/.../releases/download/v1.2.0/mobilenet_v3_small_xnnpack.pte
```

## Available Models

### Image Classification (mobilenet)

| Model | Backend | Description |
|-------|---------|-------------|
| `mobilenet_v3_small_xnnpack.pte` | XNNPACK | CPU-optimized, all platforms |
| `mobilenet_v3_small_coreml.pte` | CoreML | Apple Neural Engine |
| `mobilenet_v3_small_mps.pte` | MPS | Apple GPU (Metal) |
| `mobilenet_v3_small_vulkan.pte` | Vulkan | Cross-platform GPU |

### Object Detection (yolo)

| Model | Backends | Description |
|-------|----------|-------------|
| `yolo11n_*.pte` | xnnpack, coreml, mps, vulkan | YOLO11 Nano |
| `yolov8n_*.pte` | xnnpack, coreml, mps, vulkan | YOLOv8 Nano |
| `yolov5n_*.pte` | xnnpack, coreml, mps, vulkan | YOLOv5 Nano |

### Pose Estimation (yolo-pose)

| Model | Backends | Description |
|-------|----------|-------------|
| `yolo11n-pose_*.pte` | xnnpack, coreml, mps, vulkan | YOLO11 Pose |
| `yolov8n-pose_*.pte` | xnnpack, coreml, mps, vulkan | YOLOv8 Pose |

### Face Detection (yolo-face)

| Model | Backends | Description |
|-------|----------|-------------|
| `yolov11n-face_*.pte` | xnnpack, coreml, mps, vulkan | YOLO11 Face |
| `yolov10n-face_*.pte` | xnnpack, coreml, mps, vulkan | YOLOv10 Face |

## Backend Selection

| Backend | Best For | Platforms |
|---------|----------|-----------|
| **XNNPACK** | Universal compatibility | All platforms |
| **CoreML** | Apple Neural Engine | iOS, macOS |
| **MPS** | Apple GPU | macOS (deprecated in 1.2.0) |
| **Vulkan** | Cross-platform GPU | Android, iOS, macOS, Windows, Linux |

## CI/CD: How Models Are Published

Models are exported and published automatically via GitHub Actions (`export-models.yml`):

1. **Export** (macOS): Installs ExecuTorch, exports all models with all backends
2. **Upload**: Uploads `.pte` files and labels to GitHub Release (`v{version}`)
3. **Metadata**: Generates `index.json` with hashes/sizes/URLs, commits to repo

### Trigger Export

```bash
# Export only latest version (automatic on push to python/)
# Export all versions (manual trigger)
gh workflow run "Export Models" -f executorch_versions=all -f models=all -f backends=all

# Export specific version
gh workflow run "Export Models" -f executorch_versions=1.2.0 -f models=all -f backends=all
```

### Manual Export

```bash
cd python
pip install -r requirements.txt
pip install executorch==1.2.0

python main.py export --all --output-dir ../1.2.0
python main.py export --mobilenet --backends xnnpack coreml
```

## Version Compatibility

**Important**: Models must match the ExecuTorch runtime version.

| Runtime Version | Release Tag |
|-----------------|-------------|
| ExecuTorch 1.0.1 | `v1.0.1` |
| ExecuTorch 1.1.0 | `v1.1.0` |
| ExecuTorch 1.2.0 | `v1.2.0` |

## License

Models are provided under their original licenses:
- **MobileNet**: Apache 2.0
- **YOLO**: AGPL-3.0

See [LICENSES/](LICENSES/) for details.

---

**Repository**: https://github.com/abdelaziz-mahdy/executorch_flutter_models
