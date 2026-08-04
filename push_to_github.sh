#!/bin/bash
# Script to initialize git repo and push to GitHub
# Usage: cd Discrete-Diffusion-Forcing && bash push_to_github.sh

GITHUB_URL="https://github.com/yhpengtu-rgb/OPSM.git"
BRANCH_NAME="feature/dynamic-topk-position-selection"
COMMIT_MSG="feat: implement dynamic top-k position selection in block decoding

- Add _select_topk_positions function for logits-based position selection
- Modify student_blockwise_rollout to use dynamic top-k instead of fixed pos
- Add DMD loss support with gradient checkpointing
- Add training config for DMD on-policy distillation
- Update train.py with async pipeline and EMA LoRA support
- Update configs: batch_size=1, num_iters=20/100, max_length=50
- Add comprehensive on-policy training documentation"

echo "=== Step 1: Initialize Git Repository ==="
# Remove existing git if any (safe check)
if [ -d ".git" ]; then
    echo "Removing existing .git directory..."
    rm -rf .git
fi

git init

echo ""
echo "=== Step 2: Create New Branch ==="
git checkout -b "$BRANCH_NAME"

echo ""
echo "=== Step 3: Add Files ==="
git add -A

echo ""
echo "=== Step 4: Commit Changes ==="
git commit -m "$COMMIT_MSG"

echo ""
echo "=== Step 5: Add Remote Repository ==="
git remote add origin "$GITHUB_URL"

echo ""
echo "=== Step 6: Push to GitHub ==="
echo "Attempting to push to $BRANCH_NAME..."
echo "NOTE: You may be prompted for GitHub credentials."
echo "      Use your GitHub username and Personal Access Token."
echo ""
echo "If push fails due to authentication, run:"
echo "  git config credential.helper store"
echo "Then try the push again."
echo ""

git push -u origin "$BRANCH_NAME"

echo ""
echo "=== Done ==="
echo "Branch '$BRANCH_NAME' has been pushed to GitHub!"
echo "You can now review at: $GITHUB_URL"
echo ""
echo "To create a Pull Request, visit:"
echo "  $GITHUB_URL/tree/$BRANCH_NAME"