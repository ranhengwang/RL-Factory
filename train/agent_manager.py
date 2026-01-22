import os
import json
import zipfile
import requests
from typing import Dict, Any, Optional
from pathlib import Path
from .base_downloader import BaseDownloader

class AgentManager(BaseDownloader):
    def __init__(self, base_url: str, project_dir: str):
        self.base_url = base_url
        self.agents_dir = os.path.join(project_dir, 'agent')
        os.makedirs(self.agents_dir, exist_ok=True)
    
    def download_agent(self, agent_name: str) -> Dict[str, Any]:
        """下载并解压 agent"""
        # 1. 获取 agent 列表
        agents = self._get_agent_list()
        # 2. 查找目标 agent
        agent_info = next(
            (a for a in agents 
             if a.get('modelName') == agent_name and a.get('trainingType') == 'agent'),
            None
        )

        if not agent_info:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        # 3. 下载 agent 包
        agent_id = agent_info['id']
        download_url = self._get_download_url(agent_id)
        zip_path = os.path.join(self.agents_dir, f"{agent_name}.zip")
        
        self._download_file(download_url, zip_path)
        
        # 4. 解压并删除 zip
        extract_dir = os.path.join(self.agents_dir, agent_name)
        self._extract_zip(zip_path, extract_dir)
        os.remove(zip_path)
        
        # 5. 读取 agent 配置
        config_path = os.path.join(extract_dir, f"{agent_name}.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            agent_config = json.load(f)
        
        return agent_config
    
