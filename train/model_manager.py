import os
import json
import zipfile
import requests
from typing import Dict, Any, Optional
from .base_downloader import BaseDownloader

class ModelManager(BaseDownloader):
    def __init__(self, base_url: str, project_dir: str):
        self.base_url = base_url
        self.models_dir = os.path.join(project_dir, 'model')
        os.makedirs(self.models_dir, exist_ok=True)
    
    def download_model(self, model_id: str) -> str:
        """下载模型"""
        # 1. 获取模型信息
        model_info = self._get_model_info(model_id)
        
        # 2. 下载模型
        download_url = self._get_download_url(model_id)
        model_name = model_info.get('modelName', f'model_{model_id}')
        zip_path = os.path.join(self.models_dir, f"{model_name}_{model_info.get('version')}_{model_id}.zip")
        
        self._download_file(download_url, zip_path)
        
        # 3. 解压并删除 zip
        extract_dir = os.path.join(self.models_dir, f"{model_name}_{model_info.get('version')}_{model_id}")
        self._extract_zip(zip_path, extract_dir)
        os.remove(zip_path)
        
        return extract_dir
