import os
import json
from typing import Dict, Any, List
from train.json_2_tools import extract_tools, generate_server_file_code, _safe_filename, write_code_to_project_tools
from train.base_downloader import BaseDownloader

class MCPManager(BaseDownloader):
    def __init__(self, base_url: str, project_dir: str, project_name: str):
        self.base_url = base_url
        self.mcp_dir = os.path.join(project_dir, 'mcp')
        self.tools_dir = os.path.join(project_dir, 'projects', project_name, 'envs', 'tools')
        # self.config_dir = os.path.join(project_dir, 'projects', project_name, 'envs', 'configs')
        self.reward_dir = os.path.join(project_dir, 'projects', project_name, 'envs', 'reward')
        self.output_dir = os.path.join(project_dir, 'projects', project_name, 'output')
        # os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.mcp_dir, exist_ok=True)
        os.makedirs(self.tools_dir, exist_ok=True)
        os.makedirs(self.reward_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def download_mcp(self, mcp_name: str) -> Dict[str, Any]:
        """下载 MCP 服务器并生成工具代码"""
        # 1. 获取 MCP 列表
        mcps = self._get_agent_list()
        
        # 2. 查找目标 MCP
        mcp_info = next(
            (m for m in mcps 
             if m.get('modelName') == mcp_name and m.get('trainingType') == 'mcp'),
            None
        )
        
        if not mcp_info:
            raise ValueError(f"MCP server '{mcp_name}' not found")
        
        # 3. 下载 MCP 包
        mcp_id = mcp_info['id']
        download_url = self._get_download_url(mcp_id)
        zip_path = os.path.join(self.mcp_dir, f"{mcp_name}.zip")
        
        self._download_file(download_url, zip_path)
        
        # 4. 解压并删除 zip
        extract_dir = os.path.join(self.mcp_dir, mcp_name)
        self._extract_zip(zip_path, extract_dir)
        # os.remove(zip_path)
        

        # 5. 读取 MCP 配置
        config_path = os.path.join(extract_dir, 'mcp.json')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            mcp_config = json.load(f)

        # 6. 生成工具代码
        tools = extract_tools(mcp_config)

        if not tools:
            raise ValueError(f"MCP server '{mcp_name}' not found tools")
        code = generate_server_file_code(mcp_name, tools)
        out_file = _safe_filename(mcp_name)
        write_code_to_project_tools(self.tools_dir, out_file, code)
        
        
        return {
            'name': mcp_name,
            'path': extract_dir,
            'config': mcp_config
        }
    