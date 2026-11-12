# Surveillance Robot

An integrated surveillance pipeline for person detection, tracked video,
identity recognition, pedestrian attributes, temporal action recognition, and
deterministic alert generation.

## Components

- **SapiensID**: known/unknown identity matching against a local database.
- **OpenPAR PromptPAR**: supported clothing and carried-item attributes.
- **PoseC3D/PYSKL**: skeleton-based individual and multi-person activities.
- **YOLO pose + tracking**: person detection, stable track IDs, and best-crop selection.
- **Alert renderer**: fixed plain-text/HTML email with annotated evidence.

The upstream model projects are included as Git submodules. Project-specific
integration code is contained in `src` and does not modify upstream source.

## Clone

```bash
git clone --recurse-submodules https://github.com/ayushmohta1550/Surveillance-Robot.git
cd Surveillance-Robot
```

The Windows/CPU compatibility changes used by the local OpenPAR runner are
stored as a reproducible patch:

```bash
git -C OpenPAR apply ../patches/openpar_windows.patch
```

Model checkpoints and datasets are intentionally not stored in Git. Place the
required weights at the paths documented in `src/README.md`.

## Image workflow

```powershell
& ".\sapiensid\.conda\sapiensid\python.exe" -m src.auto_workflow `
  ".\input\images.jpg" `
  --location "Main Gate" `
  --source-id "CAM-01"
```

## Video workflow

```powershell
& ".\sapiensid\.conda\sapiensid\python.exe" -m src.video `
  ".\input\surveillance.mp4" `
  --location "Main Gate" `
  --source-id "ROBOT-CAM-01"
```

Use `--scan-only` to verify detection and persistent tracking without loading
the three recognition models.

## Repository policy

The `.gitignore` excludes environments, model weights, datasets, identity
databases, inputs, outputs, caches, credentials, and generated PDFs. Do not
commit personal surveillance data or email credentials.
