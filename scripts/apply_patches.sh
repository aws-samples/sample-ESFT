#!/bin/bash
# Apply patches to ms-swift submodule

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PATCHES_DIR="$PROJECT_ROOT/patches"
SUBMODULE_DIR="$PROJECT_ROOT/ms-swift"

echo "Applying patches to ms-swift..."

# Check if submodule exists (can be a file or directory)
if [ ! -e "$SUBMODULE_DIR/.git" ]; then
    echo "Error: ms-swift submodule not found. Please run 'git submodule update --init' first."
    exit 1
fi

# Apply the gpt_bridge patch
if [ -f "$PATCHES_DIR/ms-swift-gpt-bridge.patch" ]; then
    echo "Applying ms-swift-gpt-bridge.patch..."
    cd "$SUBMODULE_DIR"
    git apply "$PATCHES_DIR/ms-swift-gpt-bridge.patch"
    echo "✓ Patch applied successfully"
else
    echo "Error: Patch file not found at $PATCHES_DIR/ms-swift-gpt-bridge.patch"
    exit 1
fi

echo "All patches applied successfully!"
