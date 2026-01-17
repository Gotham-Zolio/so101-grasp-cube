# Install Dependencies for Real Robot Deployment

## Quick Install

Run the appropriate command for your system:

### macOS / Linux (Bash/Zsh)
```bash
bash install_dependencies.sh
```

Or:
```bash
sh install_dependencies.sh
```

### Windows (PowerShell)
```powershell
.\install_dependencies.ps1
```

## Manual Installation

If the scripts don't work, run these commands manually:

### 1. Install env_client (Required)
```bash
uv pip install -e packages/env-client
```

### 2. Install LeRobot (Required)
```bash
uv pip install lerobot
```

### 3. Install other dependencies (Optional but recommended)
```bash
uv pip install torch torchvision torchaudio
uv pip install diffusers transformers einops
```

## Verification

After installation, verify everything works:

```bash
# Test imports
uv run python -c "import env_client; print('✓ env_client installed')"
uv run python -c "import lerobot; print('✓ lerobot installed')"

# Test server can be imported
uv run python -c "from grasp_cube.real.serve_diffusion_policy import *; print('✓ serve_diffusion_policy imports OK')"
```

## If you see import errors

### "No module named 'env_client'"
```bash
uv pip install -e packages/env-client
```

### "No module named 'lerobot.utils.constants'" or "No module named 'lerobot.processor.factory'"
```bash
pip uninstall lerobot -y
uv pip install lerobot --force-reinstall
```

### "No module named 'tyro'"
```bash
uv pip install tyro
```

## Complete fresh install (if everything else fails)

```bash
# Remove old environment
uv venv --python 3.10 --force

# Reinstall everything
uv pip install -e .
uv pip install -e packages/env-client
uv pip install lerobot
uv pip install tyro
```

Then try the server again:
```bash
uv run python grasp_cube/real/serve_diffusion_policy.py --policy.path checkpoints/lift_real/checkpoint-best --policy.task lift
```
