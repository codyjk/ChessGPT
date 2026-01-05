# Cloud Deployment Testing Guide

## What Was Built

A complete cloud GPU deployment system with:

### Core Components
- **Provider Abstraction** - Supports Lambda Labs API, generic SSH, and local testing
- **Remote Execution** - SSH commands, tmux sessions, environment setup
- **Data Sync** - Smart rsync-based transfer (only small files from local machine)
- **Deployment Orchestration** - One-command end-to-end workflow
- **Instance Management** - CLI for listing, monitoring, retrieving models

### File Structure
```
scripts/cloud/
├── __init__.py           # Package init
├── providers.py          # CloudProvider base class + Lambda/SSH implementations
├── remote.py             # SSH execution, tmux, environment setup
├── sync.py               # rsync utilities for data transfer
├── deploy.py             # Main deployment CLI
├── instances.py          # Instance management CLI
└── test_local.py         # Local testing utilities

configs/cloud/
└── lambda.yaml           # Lambda Labs GPU types and pricing
```

### CLI Commands Added
- `poetry run cloud-train` - Deploy training to cloud
- `poetry run cloud-instances` - Manage instances
- `poetry run cloud-test` - Local testing utilities

---

## Local Testing (Before Cloud)

### Step 1: Enable SSH to Localhost

**macOS:**
```bash
# Enable Remote Login
# System Settings → General → Sharing → Toggle "Remote Login"

# Or set up passwordless SSH
ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""  # If you don't have a key
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Test it
ssh localhost echo "working"
```

**Linux:**
```bash
# Install SSH server
sudo apt-get install openssh-server

# Set up passwordless SSH
ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Test it
ssh localhost echo "working"
```

### Step 2: Install Dependencies

```bash
# Install deployment tools
poetry install --with model,deploy

# This installs:
# - rich (pretty terminal output)
# - requests (Lambda API)
# - pyyaml (config parsing)
```

### Step 3: Run Local Tests

```bash
# Test SSH, rsync, and command execution
poetry run cloud-test

# Or test individual components
poetry run cloud-test --test ssh      # Just SSH
poetry run cloud-test --test rsync    # Just rsync
poetry run cloud-test --test commands # Just remote commands
poetry run cloud-test --test dry-run  # Show what would happen
```

**Expected Output:**
```
============================================================
Cloud Deployment Local Testing
============================================================

Testing SSH connection to localhost...
✓ SSH to localhost working

Testing rsync...
✓ rsync working

Testing remote command execution...
✓ Basic echo: test
✓ Python version: Python 3.10.x
✓ Poetry check: /Users/you/.local/bin/poetry

============================================================
✓ All local tests passed!
============================================================
```

### Step 4: Test Full Deployment Workflow (Localhost)

This runs the entire deployment workflow but on your local machine:

```bash
# Deploy to localhost using micro config (fast test)
poetry run cloud-train \
  --provider local-test \
  --gpu test \
  --config micro_test_gpt2_medium \
  --name test-local
```

**What This Does:**
1. ✓ Creates mock instance (localhost)
2. ✓ SSH to localhost
3. ✓ Rsync project files to `~/ChessGPT`
4. ✓ Run `poetry install --with model`
5. ✓ Rsync training data (tokenizer + CSVs)
6. ✓ Start training in tmux session `chessgpt-test-local`

**Watch Training:**
```bash
# Attach to tmux session
tmux attach -t chessgpt-test-local

# Or just monitor output
tmux capture-pane -t chessgpt-test-local -p | tail -50
```

**Result:**
- Training runs on your machine (uses local GPU/CPU)
- Model saved to `~/ChessGPT/models/test-local/`
- Tests the entire workflow without cloud costs

---

## Component Testing

### Test Provider Abstraction

```python
# Test in Python REPL
from scripts.cloud.providers import get_provider

# Local test provider
provider = get_provider("local-test")
instance = provider.provision("test-gpu")
print(f"Instance: {instance.id} @ {instance.ip}")
provider.terminate(instance)
```

### Test SSH Execution

```python
from scripts.cloud.providers import get_provider, SSHConfig, Instance
from scripts.cloud.remote import ssh_exec

# Create localhost instance
ssh_config = SSHConfig(host="localhost", user="your-username", port=22)
instance = Instance(
    id="test", provider="local", gpu_type="test",
    ip="127.0.0.1", ssh_config=ssh_config, price_per_hour=0.0
)

# Test command execution
result = ssh_exec(instance, "echo 'Hello from localhost'")
print(result.stdout)

# Test GPU check
result = ssh_exec(instance, "nvidia-smi", check=False)
print("GPU available:", result.returncode == 0)
```

### Test Data Sync

```bash
# Create test directory
mkdir -p /tmp/test-sync
echo "test" > /tmp/test-sync/test.txt

# Test rsync to localhost
rsync -avz /tmp/test-sync/ localhost:/tmp/test-dest/

# Verify
ssh localhost ls /tmp/test-dest/
```

---

## Verification Checklist

Before deploying to actual cloud GPU:

- [ ] SSH to localhost works: `ssh localhost echo "test"`
- [ ] Local tests pass: `poetry run cloud-test`
- [ ] Full localhost deployment works: `poetry run cloud-train --provider local-test --gpu test --config micro_test_gpt2_medium --name test`
- [ ] Training starts in tmux: `tmux list-sessions` shows `chessgpt-test-local`
- [ ] Model saved locally: Check `~/ChessGPT/models/test-local/`

---

## Next: Cloud Deployment

Once local tests pass, you're ready for actual cloud deployment:

### Option 1: Lambda Labs (Easiest)

1. Get API key: https://cloud.lambdalabs.com
2. Set env: `export LAMBDA_API_KEY=your_key`
3. Generate SSH key: `ssh-keygen -t ed25519 -f ~/.ssh/chessgpt_deploy`
4. Upload public key to Lambda dashboard
5. Deploy:
   ```bash
   poetry run cloud-train --provider lambda --gpu A100 --config phase1_gpt2_medium --name gpt2-cloud-v1
   ```

### Option 2: SSH Provider (RunPod, Vast.ai, Custom)

1. Manually provision GPU instance on provider
2. Note SSH details (IP, user, key path)
3. Deploy:
   ```bash
   poetry run cloud-train --provider ssh --gpu "RTX 4090" --config phase1_gpt2_medium --name gpt2-cloud-v1
   # Enter SSH details when prompted
   ```

---

## Troubleshooting

### SSH to localhost fails
```bash
# macOS: Enable Remote Login
System Settings → General → Sharing → Remote Login

# Or set up passwordless SSH
ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### rsync fails
```bash
# Make sure SSH to localhost works first
ssh localhost echo "test"

# Try manual rsync
rsync -avz README.md localhost:/tmp/test.md
```

### Training doesn't start
```bash
# Check tmux session exists
tmux list-sessions

# View logs
tmux capture-pane -t chessgpt-<model-name> -p

# Or attach interactively
tmux attach -t chessgpt-<model-name>
```

### Poetry not found in remote
```bash
# SSH to localhost and verify
ssh localhost 'which poetry'

# If not found, install
ssh localhost 'curl -sSL https://install.python-poetry.org | python3 -'
```

---

## What Gets Tested

| Component | Local Test | Cloud Test |
|-----------|------------|------------|
| Provider provisioning | ✓ (mock) | ✓ (real API) |
| SSH connection | ✓ (localhost) | ✓ (cloud IP) |
| rsync file transfer | ✓ (localhost) | ✓ (cloud) |
| Environment setup | ✓ (local) | ✓ (cloud) |
| Poetry installation | ✓ (local) | ✓ (cloud) |
| Data sync | ✓ (local) | ✓ (cloud) |
| Training execution | ✓ (local) | ✓ (cloud) |
| Tmux persistence | ✓ (local) | ✓ (cloud) |
| Cost tracking | ✓ ($0) | ✓ (actual) |

The local tests validate the entire workflow without cloud costs!
