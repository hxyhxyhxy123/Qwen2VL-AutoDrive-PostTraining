# GitHub Upload Guide

This folder is prepared as a clean GitHub project. It keeps source code,
configuration files, scripts, curated metrics, and documentation. Large
datasets, checkpoints, cache files, and generated predictions should not be
committed.

## 1. Create a GitHub Repository

Recommended repository name:

```text
Qwen2VL-AutoDrive-PostTraining
```

On GitHub, create an empty public repository without adding a README,
`.gitignore`, or license, because this project already contains them.

## 2. Initialize Git Locally

Run these commands in PowerShell:

```powershell
cd E:\Algorithm\Qwen2VL-AutoDrive-PostTraining
git init
git add .
git status
git commit -m "Initial release: AutoDrive VLM post-training pipeline"
```

## 3. Connect the Remote

Replace `<YOUR_GITHUB_USERNAME>` with your GitHub account name:

```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/Qwen2VL-AutoDrive-PostTraining.git
git push -u origin main
```

## 4. If GitHub Asks for Login

Use a GitHub Personal Access Token instead of your password. The token needs
repository write permission. Do not paste the token into README files, scripts,
or screenshots.

## 5. Check Before Pushing

Make sure these are not included:

```text
data/
checkpoints/
model_cache/
*.safetensors
*.bin
outputs/
```

The curated public metrics are stored in:

```text
results/
docs/
```

