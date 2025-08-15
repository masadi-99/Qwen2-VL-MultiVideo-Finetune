#!/bin/bash

# Setup script for new multi-video repository
# Run this after creating your new repository on GitHub

echo "=== Qwen2-VL Multi-Video Repository Setup ==="
echo ""

# Get the new repository URL from user
read -p "Enter your new GitHub repository URL (e.g., https://github.com/YOUR_USERNAME/Qwen2-VL-MultiVideo-Finetune.git): " NEW_REPO_URL

if [ -z "$NEW_REPO_URL" ]; then
    echo "Error: Repository URL cannot be empty"
    exit 1
fi

echo ""
echo "Setting up repository with URL: $NEW_REPO_URL"
echo ""

# Remove old remote
echo "1. Removing old remote..."
git remote remove origin

# Add new remote
echo "2. Adding new remote..."
git remote add origin "$NEW_REPO_URL"

# Push to new repository
echo "3. Pushing code to new repository..."
git push -u origin master

echo ""
echo "✅ Setup complete!"
echo ""
echo "Your enhanced multi-video Qwen2-VL repository is now available at:"
echo "$NEW_REPO_URL"
echo ""
echo "To clone on your server:"
echo "git clone $NEW_REPO_URL"
echo ""
echo "To start training with 30+ videos:"
echo "bash scripts/finetune_multivideo.sh"
echo ""