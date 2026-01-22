import os
import shutil
from typing import Optional

def ensure_dir(path: str) -> str:
    """确保目录存在，如果不存在则创建"""
    os.makedirs(path, exist_ok=True)
    return path

def remove_file(path: str) -> bool:
    """删除文件，如果存在"""
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        return True
    except Exception:
        return False

def get_file_extension(url: str) -> str:
    """从 URL 中获取文件扩展名"""
    return os.path.splitext(url)[1] or '.bin'