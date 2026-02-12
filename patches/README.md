# MS-Swift Patches

This directory contains patches for the ms-swift submodule.

## Overview

The ms-swift submodule is based on commit `f626485c31d5fac1c95efea39e46e746439ac488` from the original repository:
https://github.com/modelscope/ms-swift.git

## Patches

### ms-swift-gpt-bridge.patch

This patch modifies `swift/megatron/model/gpt_bridge.py` to fix SequentialMLP saving errors when pipeline parallelism (PP) > 1.

**Key changes:**
- Adds `_skip_pp_broadcast` flag to skip collective operations during SequentialMLP saving
- Implements `_set_mlp_state_sequential()` method for proper SequentialMLP handling
- Adds support for SequentialMLP loading with expert distribution
- Fixes synchronization issues across EP-PP ranks

## Usage

### Initial Setup

1. Initialize the submodule:
```bash
git submodule update --init --recursive
```

2. Apply patches:
```bash
./scripts/apply_patches.sh
```

### Updating the Submodule

If you need to update to a newer version of ms-swift:

1. Update the submodule to the desired commit:
```bash
cd ms-swift
git fetch upstream
git checkout <new-commit-hash>
cd ..
```

2. Regenerate the patch if needed:
```bash
cd ms-swift
git diff <base-commit> HEAD swift/megatron/model/gpt_bridge.py > ../patches/ms-swift-gpt-bridge.patch
cd ..
```

3. Update this README with the new base commit hash.

## Maintenance

- Keep patches minimal and focused
- Document all changes in this README
- Test patches after regenerating them
- Consider upstreaming patches to the original repository when appropriate
