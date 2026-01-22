import os
import json
from typing import Dict, Any

class ConfigGenerator:
    def __init__(self, project_dir: str, project_name: str):
        self.project_dir = project_dir
        self.project_name = project_name 
        self.config_dir = os.path.join(project_dir, 'projects', project_name, 'envs', 'configs')
        os.makedirs(self.config_dir, exist_ok=True)
    
    def generate_config(self, agent_name: str, mcp_servers: Dict[str, Any]) -> str:
        """生成 MCP 配置文件"""
        config = {
            'mcpServers': {}
        }

        project_root = os.path.join(self.project_dir, 'projects', self.project_name)
        tools_dir = os.path.join(project_root, 'envs', 'tools')
        
        # 为每个 MCP 服务器添加配置
        for mcp_name, mcp_info in mcp_servers.items():
            server_py = os.path.join(tools_dir, f"{mcp_name}.py")
            server_py_abs = os.path.abspath(server_py)

            if not os.path.exists(server_py_abs):
                raise FileNotFoundError(f"找不到 MCP server 代码文件: {server_py_abs}")

            config['mcpServers'][mcp_name] = {
                'command': 'python3',
                'args': [server_py_abs]
            }
        
        # 写入文件
        config_path = os.path.join(self.config_dir, f"{agent_name}.pydata")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump([config], f, indent=4, ensure_ascii=False)
        
        return config_path