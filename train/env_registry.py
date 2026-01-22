import os
import re
import traceback
from typing import Optional


class EnvRegistry:
    """环境注册器 - 负责将环境自动注册到 envs/__init__.py"""
    
    def __init__(self, project_dir: str):
        """
        初始化环境注册器
        
        :param project_dir: 项目目录路径 (通常是 train/ 目录)
        """
        self.project_dir = project_dir
        self.root_dir = os.path.dirname(project_dir)
        print(f"[EnvRegistry] 项目目录: {self.project_dir}")
        print(f"[EnvRegistry] 根目录: {self.root_dir}")
        self.init_file_path = os.path.join(self.root_dir, "envs", "__init__.py")
    
    def register(self, agent_name: str) -> bool:
        """
        注册环境到 envs/__init__.py
        
        :param agent_name: agent 名称
        :return: 注册是否成功
        """
        # 1. 验证文件存在性
        if not self._validate_files(agent_name):
            return False
        
        # 2. 生成类名和导入语句
        print("============="+"开始注册环境"+"=============")
        class_name = self._generate_class_name(agent_name)
        import_stmt = self._generate_import_statement(agent_name, class_name)
        
        print(f"\n[Auto-Register] 正在注册环境到 {self.init_file_path} ...")
        
        try:
            # 3. 读取现有内容
            content = self._read_init_file()
            
            # 4. 检查是否已注册
            if self._is_already_registered(content, import_stmt):
                print(f"✅ 环境 {agent_name} 已注册 (Import exists)。")
                return True
            
            # 5. 修改文件内容
            content = self._add_import_statement(content, import_stmt)
            content = self._update_all_list(content, class_name)
            content = self._update_registry_dict(content, agent_name, class_name)
            
            # 6. 写回文件
            self._write_init_file(content)
            
            print(f"✅ 注册成功: {class_name} -> {self.init_file_path}")
            print(f"   Import: {import_stmt}")
            print(f"   Registry: '{agent_name}' -> {class_name}")
            return True
            
        except Exception as e:
            print(f"❌ 注册失败: {e}")
            traceback.print_exc()
            return False
    
    def _validate_files(self, agent_name: str) -> bool:
        """验证必要的文件是否存在"""
        env_file_path = os.path.join(
            self.project_dir, "projects", agent_name, 
            "envs", "reward", f"{agent_name}.py"
        )
        
        if not os.path.exists(env_file_path):
            with open(env_file_path, 'w', encoding='utf-8') as f:
                f.write("")
            print(f" 找不到奖励函数文件: {env_file_path}，已创建空文件，跳过注册。")
        
        return True
    
    def _generate_class_name(self, agent_name: str) -> str:
        """
        将 agent 名称转换为类名
        例如: my_weather_query -> MyWeatherQueryEnv
        """
        return "".join([word.capitalize() for word in agent_name.split('_')]) + "Env"
    
    def _generate_import_statement(self, agent_name: str, class_name: str) -> str:
        """生成导入语句"""
        import_path = f"train.projects.{agent_name}.envs.reward.{agent_name}"
        return f"from {import_path} import {class_name}"
    
    def _read_init_file(self) -> str:
        """读取 __init__.py 文件内容"""
        with open(self.init_file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _write_init_file(self, content: str) -> None:
        """写入 __init__.py 文件内容"""
        with open(self.init_file_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    def _is_already_registered(self, content: str, import_stmt: str) -> bool:
        """检查是否已经注册"""
        return import_stmt in content
    
    def _add_import_statement(self, content: str, import_stmt: str) -> str:
        """添加导入语句到文件中"""
        if "__all__" not in content:
            return f"{import_stmt}\n{content}"
        
        # 找到最后一个 import 语句的位置，在其后插入
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in content.split('\n'):
            if line.strip().startswith('from ') or line.strip().startswith('import '):
                import_lines.append(line)
            elif line.strip().startswith('#') or line.strip() == '':
                if in_imports:
                    import_lines.append(line)
                else:
                    other_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        
        import_lines.append(import_stmt)
        return '\n'.join(import_lines) + '\n' + '\n'.join(other_lines)
    
    def _update_all_list(self, content: str, class_name: str) -> str:
        """更新 __all__ 列表"""
        match_all = re.search(r"(__all__\s*=\s*\[)([^\]]*?)(\])", content, re.DOTALL)
        if not match_all:
            return content
        
        current_list = match_all.group(2)
        if class_name in current_list:
            return content
        
        # 追加到列表末尾，保持格式
        if current_list.strip().endswith(','):
            new_list = f"{current_list} '{class_name}'"
        else:
            new_list = f"{current_list}, '{class_name}'"
        
        return content.replace(
            match_all.group(0), 
            f"{match_all.group(1)}{new_list}{match_all.group(3)}"
        )
    
    def _update_registry_dict(self, content: str, agent_name: str, class_name: str) -> str:
        """更新 TOOL_ENV_REGISTRY 字典"""
        match_reg = re.search(r"(TOOL_ENV_REGISTRY\s*=\s*\{)([^\}]*?)(\})", content, re.DOTALL)
        if not match_reg:
            return content
        
        current_dict = match_reg.group(2)
        
        # 检查 key 是否已存在
        if f"'{agent_name}'" in current_dict or f'"{agent_name}"' in current_dict:
            return content
        
        new_entry = f",\n    '{agent_name}': {class_name}"
        return content.replace(
            match_reg.group(0),
            f"{match_reg.group(1)}{current_dict}{new_entry}{match_reg.group(3)}"
        )