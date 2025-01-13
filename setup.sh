#!/bin/bash

# 检查是否存在 requirements.txt 文件
if [ -f "requirements.txt" ]; then
  echo "Found requirements.txt. Installing dependencies..."
  # 安装 requirements.txt 中的依赖
  pip install -r requirements.txt
  
  echo "Dependencies installed successfully."
else
  echo "requirements.txt not found in the current directory."
fi