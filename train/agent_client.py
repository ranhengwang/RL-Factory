import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from train.agent_manager import AgentManager
from train.model_manager import ModelManager
from train.mcp_manager import MCPManager
from train.base_downloader import BaseDownloader
from train.config_generator import ConfigGenerator
from train.env_registry import EnvRegistry
import logging
from train.process_trajectory import process_api_items, save_parquet, fetch_from_api
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from train.training_script_template import get_training_script_template
import uvicorn
import time
import json
from dotenv import load_dotenv
from train.prompt import get_reward_generation_prompt, get_system_prompt
import requests
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # 添加这行
    max_age=3600,  # 添加这行,缓存预检请求
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STORE_URL = "http://192.168.1.86:8080/api/v1"
TRAJECTORY_URL = "http://192.168.1.86:8224"

PROJECT_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train", "projects")

class CreateProjectRequest(BaseModel):
    agent_name: str

class AgentClient:
    def __init__(self, base_url: str, project_dir: str = None, project_name: str = None):
        self.base_url = base_url
        self.project_dir = project_dir or os.getcwd()
        self.agent_manager = AgentManager(base_url, self.project_dir)
        self.model_manager = ModelManager(base_url, self.project_dir)
        self.mcp_manager = MCPManager(base_url, self.project_dir, project_name)
        self.config_generator = ConfigGenerator(self.project_dir, project_name)
        self.env_registry = EnvRegistry(self.project_dir)
        self.project_name = project_name
        self.instruction = None
        # self.train_file = None
        # self.val_file = None
        self.config_path = None
        self.model_path = None

    # @app.get("/download_agent/{agent_name}")
    def download_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        下载并设置 agent
        
        1. 下载 agent 压缩包并解压
        2. 下载关联的模型
        3. 下载所有 MCP 服务器
        4. 生成配置文件
        
        :param agent_name: 要下载的 agent 名称
        :return: 包含下载和设置结果的字典
        """
        try:
            # 1. 下载 agent
            agent_info = self.agent_manager.download_agent(agent_name)
            if not agent_info:
                raise ValueError(f"Failed to download agent: {agent_name}")
            
            # logger.debug(f"Downloaded agent info: {agent_info}")
            
            self.instruction = agent_info['agent_config']['instruction']
            print("============="+"函数内 agent 完成"+"=============")
            # 2. 下载模型
            model_path = None
            if 'agent_config' in agent_info and 'model_id' in agent_info['agent_config']:
                print("============="+"开始下载模型"+"=============")
                print(f"Downloading model ID: {agent_info['agent_config']['model_id']}")
                # 返回下载解压目录
                model_path = self.model_manager.download_model(agent_info['agent_config']['model_id'])
                self.model_path = model_path
                # logger.debug(f"Downloaded model path: {model_path}")
            print("============="+"下载模型完成"+"=============")
            # 3. 下载 MCP 服务器
            mcp_servers = {}
            if 'agent_config' in agent_info and 'mcp' in agent_info['agent_config']:
                for mcp_name in agent_info['agent_config']['mcp']:
                    mcp_info = self.mcp_manager.download_mcp(mcp_name)
                    if mcp_info:
                        mcp_servers[mcp_name] = mcp_info
            
            # 4. 生成配置文件
            config_path = self.config_generator.generate_config(agent_name, mcp_servers)
            self.config_path = config_path
            return {
                'status': 'success',
                'agent': agent_info,
                'model_path': model_path,
                'mcp_servers': mcp_servers,
                'config_path': config_path
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    # @app.get("/build_parquet/{task_type}/{batch_size}")
    def build_parquet(self, task_type: str, batch_size: int = 16) -> Dict[str, Any]:
        api_items = fetch_from_api(TRAJECTORY_URL, task_type, batch_size, timeout=60)
        if not api_items:
            raise ValueError(f"Failed to fetch API items: {task_type}")
        
        data = process_api_items(api_items, self.instruction)
        if not data:
            raise ValueError(f"Failed to process API items: {task_type}")
        
        out_path = os.path.join(self.project_dir, 'projects', self.project_name, 'data', f"{task_type}.parquet")
        save_parquet(data, out_path)
        # self.train_file = out_path
        # self.val_file = out_path
        return out_path

    def register_env_to_registry(self, agent_name):
        """
        自动注册环境到 envs/__init__.py
        """
        print("============="+"进入注册环境"+"=============")
        success = self.env_registry.register(agent_name)
        return success
    
    def generate_training_script(self, agent_name):
        """
        生成训练脚本 (.sh)
        """
        script_path = os.path.join(self.project_dir, "projects", agent_name, f"{agent_name}.sh")
        result_dir = os.path.join(self.project_dir, "projects", agent_name, "output")
        script_content = get_training_script_template(
            agent_name=agent_name,
            model_path=self.model_path,
            train_file=os.path.join(self.project_dir, 'projects', self.project_name, 'data', f"{agent_name}.parquet"),
            val_file=os.path.join(self.project_dir, 'projects', self.project_name, 'data', f"{agent_name}.parquet"),
            config_path=self.config_path,
            result_dir=result_dir
        )
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            print(f"\n✅ 训练脚本已生成: {script_path}")
            return script_path
        except Exception as e:
            print(f"\n❌ 生成训练脚本失败: {e}")
            return None


# ============== FastAPI 路由 ==============

@app.get("/rl/agents")
async def get_agent_list() -> Dict[str, List[Dict[str, Any]]]:
    """
    获取所有可用的 agent 列表
    """
    try:
        # 创建临时的 BaseDownloader 实例来获取列表
        agent_manager = BaseDownloader(STORE_URL)
        agents = agent_manager._get_agent_list()
        
        # 过滤出 agent 类型的模型
        agent_list = []
        for agent in agents:
            if agent.get('trainingType') == 'agent':
                agent_list.append({
                    'id': agent.get('id'),
                    'name': agent.get('modelName'),
                    'description': agent.get('description', ''),
                    'created_at': agent.get('createdAt', '')
                })
        
        return {"agents": agent_list}
    except Exception as e:
        logger.error(f"获取 agent 列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取 agent 列表失败: {str(e)}")


@app.post("/rl/projects/create")
async def create_project(request: CreateProjectRequest) -> Dict[str, Any]:
    """
    创建新项目：
    1. 检查项目是否已存在
    2. 下载 agent 及其依赖
    3. 创建项目目录结构
    """
    agent_name = request.agent_name.strip()
    
    if not agent_name:
        raise HTTPException(status_code=400, detail="Agent 名称不能为空")
    
    # 检查项目是否已存在
    project_path = Path(PROJECT_BASE_DIR) / agent_name
    if project_path.exists():
        raise HTTPException(
            status_code=400, 
            detail=f"项目 '{agent_name}' 已存在，无法重复创建"
        )
    
    try:
        # 创建 AgentClient 实例并下载 agent
        agent_client = AgentClient(
            base_url=STORE_URL,
            project_name=agent_name
        )
        
        # 下载 agent 及其依赖
        result = agent_client.download_agent(agent_name)
        print("============="+"下载 agent 完成"+"=============")
        if result['status'] != 'success':
            raise HTTPException(
                status_code=500,
                detail=f"下载 agent 失败: {result.get('message', 'Unknown error')}"
            )
        print("============="+"开始注册环境"+"=============")
        # 注册环境
        agent_client.register_env_to_registry(agent_name)

        print("============="+"注册环境完成"+"=============")
        
        # 生成训练脚本
        script_path = agent_client.generate_training_script(agent_name)
        
        return {
            "status": "success",
            "message": f"项目 '{agent_name}' 创建成功",
            "project_name": agent_name,
            "project_path": str(project_path),
            "script_path": script_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建项目失败: {e}")
        # 清理失败的项目目录
        if project_path.exists():
            import shutil
            shutil.rmtree(project_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"创建项目失败: {str(e)}")

@app.get("/rl/projects")
async def get_project_list() -> List[Dict[str, Any]]:
    """
    获取 train/projects 目录下实际存在的项目列表
    """
    try:
        projects_dir = Path(PROJECT_BASE_DIR)
        
        if not projects_dir.exists():
            logger.warning(f"项目目录不存在: {projects_dir}")
            return []
        
        projects = []
        
        # 遍历 projects 目录下的所有子目录
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            
            # 跳过隐藏目录和临时目录
            if project_dir.name.startswith('.') or project_dir.name.startswith('__'):
                continue
            
            project_name = project_dir.name
            
            # 获取项目创建时间（从目录创建时间）
            try:
                stat = project_dir.stat()
                created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime))
            except Exception:
                created_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 检查训练状态
            is_running = False
            pid_file = project_dir / "training.pid"
            
            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    
                    # 检查进程是否存在且正在运行
                    try:
                        os.kill(pid, 0)  # 发送信号 0 检查进程是否存在
                        
                        # 进一步检查进程状态（Linux/Unix）
                        if hasattr(os, 'waitpid'):
                            try:
                                # 使用 WNOHANG 非阻塞检查进程状态
                                result = os.waitpid(pid, os.WNOHANG)
                                if result[0] == 0:
                                    # 进程仍在运行
                                    is_running = True
                                else:
                                    # 进程已退出，清理 PID 文件
                                    logger.info(f"进程 {pid} 已退出，清理 PID 文件: {pid_file}")
                                    pid_file.unlink(missing_ok=True)
                                    is_running = False
                            except ChildProcessError:
                                # 不是子进程，只能通过 kill(0) 判断
                                is_running = True
                        else:
                            # Windows 或其他系统，假设进程存在即为运行中
                            is_running = True
                            
                    except OSError:
                        # 进程不存在，清理 PID 文件
                        logger.info(f"进程 {pid} 不存在，清理 PID 文件: {pid_file}")
                        pid_file.unlink(missing_ok=True)
                        is_running = False
                        
                except (ValueError, IOError) as e:
                    logger.warning(f"读取 PID 文件失败: {pid_file}, {e}")
                    is_running = False
            
            # 检查是否有历史运行记录（output 目录下的 checkpoint）
            has_runs = False
            output_dir = project_dir / "output"
            if output_dir.exists():
                checkpoint_dirs = list(output_dir.glob("global_step_*"))
                has_runs = len(checkpoint_dirs) > 0
            
            # 统计项目信息
            projects.append({
                "id": project_name,
                "name": project_name,
                "created": created_time,
                "runs": 1 if has_runs else 0,
                "running": 1 if is_running else 0,
                "latest_run_id": None,
                "empty": not has_runs,
                "tags": []
            })
        
        # 按创建时间降序排序
        projects.sort(key=lambda x: x.get("created", ""), reverse=True)
        
        logger.info(f"找到 {len(projects)} 个项目")
        return projects
    
    except Exception as e:
        logger.error(f"读取项目列表失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"读取项目列表失败: {str(e)}")


@app.delete("/rl/projects/{project_id}")
async def delete_project(project_id: str) -> Dict[str, Any]:
    """
    删除项目（物理目录）
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        # 删除物理目录
        import shutil
        shutil.rmtree(project_path)
        
        logger.info(f"已删除项目: {project_path}")
        
        return {
            "status": "ok",
            "message": f"项目 '{project_id}' 已删除",
            "deleted_project": project_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除项目失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除项目失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}

@app.get("/rl/projects/{project_id}/data")
async def get_project_data(project_id: str, batch_size: int = 16) -> Dict[str, Any]:
    """
    获取项目的原始训练数据（从 TRAJECTORY_URL）
    
    Args:
        project_id: 项目ID
        batch_size: 获取数据的条数，默认16条
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        # 从 TRAJECTORY_URL 获取原始数据
        task_type = project_id
        
        logger.info(f"正在从 {TRAJECTORY_URL} 获取项目 {project_id} 的数据，数量: {batch_size}...")
        
        try:
            api_items = fetch_from_api(TRAJECTORY_URL, task_type, batch_size, timeout=60)
            if not api_items:
                raise ValueError(f"未能从 API 获取数据")
            
            return {
                "status": "success",
                "project_id": project_id,
                "task_type": task_type,
                "data": api_items,
                "count": len(api_items) if isinstance(api_items, list) else 1,
                "batch_size": batch_size
            }
        except Exception as e:
            logger.error(f"从 API 获取数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取项目数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取项目数据失败: {str(e)}")


@app.post("/rl/projects/{project_id}/data")
async def save_project_data(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    保存编辑后的项目数据到临时文件
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        data = payload.get('data')
        task_type = payload.get('task_type', project_id)
        
        if not data:
            raise HTTPException(status_code=400, detail="数据内容不能为空")
        
        # 保存到临时目录
        temp_dir = project_path / "temp_data"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file = temp_dir / f"{task_type}_edited.json"
        
        import json
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存编辑后的数据到: {temp_file}")
        
        return {
            "status": "success",
            "message": "数据保存成功",
            "project_id": project_id,
            "temp_file": str(temp_file),
            "record_count": len(data) if isinstance(data, list) else 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存项目数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存项目数据失败: {str(e)}")


@app.post("/rl/projects/{project_id}/generate-parquet")
async def generate_project_parquet(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据编辑后的数据生成 parquet 文件
    参考 build_parquet 的实现
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        task_type = payload.get('task_type', project_id)
        
        # 读取编辑后的数据
        temp_dir = project_path / "temp_data"
        temp_file = temp_dir / f"{task_type}_edited.json"
        
        if not temp_file.exists():
            raise HTTPException(
                status_code=404, 
                detail="未找到编辑后的数据，请先保存数据"
            )
        
        import json
        with open(temp_file, 'r', encoding='utf-8') as f:
            api_items = json.load(f)
        
        # 创建 AgentClient 实例来处理数据
        # 需要先加载 instruction
        agent_client = AgentClient(
            base_url=STORE_URL,
            project_dir=os.path.dirname(PROJECT_BASE_DIR),
            project_name=project_id
        )
        
        # 读取项目的 instruction（从已下载的 agent 配置）
        config_file = project_path / "envs" / "configs" / f"{project_id}.pydata"
        if config_file.exists():
            try:
                import pickle
                with open(config_file, 'rb') as f:
                    config = pickle.load(f)
                    agent_client.instruction = config.get('instruction', '')
            except Exception as e:
                logger.warning(f"无法加载 instruction: {e}")
                agent_client.instruction = ""
        
        # 处理数据
        if not agent_client.instruction:
            logger.warning("未找到 instruction，使用空字符串")
            agent_client.instruction = ""
        
        data = process_api_items(api_items, agent_client.instruction)
        if not data:
            raise ValueError("数据处理失败")
        
        # 保存为 parquet 文件
        data_dir = project_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = data_dir / f"{task_type}.parquet"
        save_parquet(data, str(out_path))
        
        logger.info(f"已生成 Parquet 文件: {out_path}")
        
        return {
            "status": "success",
            "message": "Parquet 文件生成成功",
            "project_id": project_id,
            "output_path": str(out_path),
            "record_count": len(data) if isinstance(data, list) else 1,
            "task_type": task_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成 Parquet 文件失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成 Parquet 文件失败: {str(e)}")

@app.get("/rl/projects/{project_id}/reward")
async def get_project_reward(project_id: str) -> Dict[str, Any]:
    """
    获取项目的奖励函数代码
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        # 查找 reward 文件
        reward_dir = project_path / "envs" / "reward"
        if not reward_dir.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 的 reward 目录不存在")
        
        # 查找 .py 文件（通常是项目名.py）
        reward_file = reward_dir / f"{project_id}.py"
        if not reward_file.exists():
            # 尝试查找目录下的第一个 .py 文件
            py_files = list(reward_dir.glob("*.py"))
            if not py_files:
                raise HTTPException(status_code=404, detail=f"未找到奖励函数文件")
            reward_file = py_files[0]
        
        # 读取文件内容
        try:
            with open(reward_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取奖励函数文件失败: {str(e)}")
        
        logger.info(f"成功读取奖励函数: {reward_file}")
        
        return {
            "status": "success",
            "project_id": project_id,
            "file_name": reward_file.name,
            "file_path": str(reward_file),
            "content": content,
            "line_count": len(content.splitlines())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取奖励函数失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取奖励函数失败: {str(e)}")


@app.post("/rl/projects/{project_id}/reward")
async def save_project_reward(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    保存项目的奖励函数代码
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        content = payload.get('content')
        if content is None:
            raise HTTPException(status_code=400, detail="缺少 content 参数")
        
        # 查找 reward 文件
        reward_dir = project_path / "envs" / "reward"
        if not reward_dir.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 的 reward 目录不存在")
        
        # 查找要保存的文件
        reward_file = reward_dir / f"{project_id}.py"
        if not reward_file.exists():
            # 尝试查找目录下的第一个 .py 文件
            py_files = list(reward_dir.glob("*.py"))
            if not py_files:
                # 如果没有文件，创建新文件
                reward_file = reward_dir / f"{project_id}.py"
            else:
                reward_file = py_files[0]
        
        # 备份原文件
        if reward_file.exists():
            backup_file = reward_file.with_suffix('.py.bak')
            try:
                import shutil
                shutil.copy2(reward_file, backup_file)
                logger.info(f"已备份原文件到: {backup_file}")
            except Exception as e:
                logger.warning(f"备份文件失败: {e}")
        
        # 保存新内容
        try:
            with open(reward_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存奖励函数文件失败: {str(e)}")
        
        logger.info(f"成功保存奖励函数: {reward_file}")
        
        return {
            "status": "success",
            "message": "奖励函数保存成功",
            "project_id": project_id,
            "file_name": reward_file.name,
            "file_path": str(reward_file),
            "line_count": len(content.splitlines()),
            "backup_created": reward_file.with_suffix('.py.bak').exists()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存奖励函数失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存奖励函数失败: {str(e)}")

@app.post("/rl/projects/{project_id}/reward/generate")
async def generate_reward_with_llm(project_id: str) -> Dict[str, Any]:
    """
    使用大模型自动生成奖励函数
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        # 1. 读取 agent 配置 JSON
        agent_json_path = Path(__file__).parent / "agent" / project_id / f"{project_id}.json"
        if not agent_json_path.exists():
            raise HTTPException(status_code=404, detail=f"未找到 agent 配置文件: {agent_json_path}")
        
        try:
            with open(agent_json_path, 'r', encoding='utf-8') as f:
                agent_config = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取 agent 配置失败: {str(e)}")
        
        # 2. 读取 MCP 工具代码
        tools_dir = project_path / "envs" / "tools"
        if not tools_dir.exists():
            raise HTTPException(status_code=404, detail=f"未找到 tools 目录: {tools_dir}")
        
        # 查找 .py 文件（排除 __init__.py）
        tool_files = [f for f in tools_dir.glob("*.py") if f.name != "__init__.py"]
        if not tool_files:
            raise HTTPException(status_code=404, detail="未找到工具文件")
        
        tool_code = ""
        for tool_file in tool_files:
            try:
                with open(tool_file, 'r', encoding='utf-8') as f:
                    tool_code += f"# File: {tool_file.name}\n"
                    tool_code += f.read()
                    tool_code += "\n\n"
            except Exception as e:
                logger.warning(f"读取工具文件 {tool_file} 失败: {e}")
        
        if not tool_code:
            raise HTTPException(status_code=500, detail="无法读取工具代码")
        
        # 3. 读取参考奖励函数
        ref_reward_path = Path(__file__).parent / "ref_reward.py"
        if not ref_reward_path.exists():
            raise HTTPException(status_code=404, detail=f"未找到参考奖励函数: {ref_reward_path}")
        
        try:
            with open(ref_reward_path, 'r', encoding='utf-8') as f:
                ref_reward_code = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取参考奖励函数失败: {str(e)}")
        
        # 4. 加载 .env 配置
        env_path = Path(__file__).parent / ".env"
        load_dotenv(env_path)
        
        llm_binding = os.getenv("LLM_BINDING", "openai")
        llm_model = os.getenv("LLM_MODEL", "gpt-4")
        llm_host = os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
        llm_api_key = os.getenv("LLM_BINDING_API_KEY", "")
        
        if not llm_api_key:
            raise HTTPException(status_code=500, detail="未配置 LLM_BINDING_API_KEY")
        
        # 5. 构建提示词
        user_prompt = get_reward_generation_prompt(agent_config, tool_code, ref_reward_code)
        system_prompt = get_system_prompt()
        
        logger.info(f"开始调用大模型生成奖励函数...")
        logger.debug(f"LLM 配置: {llm_binding}, {llm_model}, {llm_host}")
        
        # 6. 调用大模型（禁用代理）
        headers = {
            "Authorization": f"Bearer {llm_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 8000
        }
        
        try:
            # 创建一个 session 并禁用代理
            session = requests.Session()
            session.trust_env = False  # 禁用环境变量中的代理设置
            
            # 明确设置 proxies 为空字典，强制禁用代理
            proxies = {
                'http': None,
                'https': None,
            }
            
            response = session.post(
                f"{llm_host}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
                proxies=proxies  # 显式禁用代理
            )
            
            if not response.ok:
                error_detail = response.text
                logger.error(f"LLM API 调用失败: {error_detail}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"LLM API 调用失败: {response.status_code} - {error_detail}"
                )
            
            result = response.json()
            generated_code = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not generated_code:
                raise HTTPException(status_code=500, detail="大模型返回为空")
            
        except requests.RequestException as e:
            logger.error(f"LLM API 请求异常: {e}")
            raise HTTPException(status_code=500, detail=f"LLM API 请求失败: {str(e)}")
        finally:
            session.close()
        
        # 7. 清理代码（移除 markdown 代码块标记）
        generated_code = generated_code.strip()
        if generated_code.startswith("```python"):
            generated_code = generated_code[len("```python"):].strip()
        if generated_code.startswith("```"):
            generated_code = generated_code[3:].strip()
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3].strip()
        
        # 8. 保存生成的代码（先保存到临时文件）
        reward_dir = project_path / "envs" / "reward"
        reward_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file = reward_dir / f"{project_id}_generated.py"
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(generated_code)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存生成的代码失败: {str(e)}")
        
        logger.info(f"奖励函数生成成功: {temp_file}")
        
        return {
            "status": "success",
            "message": "奖励函数生成成功",
            "project_id": project_id,
            "generated_file": str(temp_file),
            "code": generated_code,
            "line_count": len(generated_code.splitlines()),
            "model_used": llm_model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成奖励函数失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成奖励函数失败: {str(e)}")

@app.get("/rl/projects/{project_id}/training-config")
async def get_training_config(project_id: str) -> Dict[str, Any]:
    """
    获取训练脚本的当前配置
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        # 读取训练脚本
        script_path = project_path / f"{project_id}.sh"
        if not script_path.exists():
            raise HTTPException(status_code=404, detail=f"未找到训练脚本: {script_path}")
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取训练脚本失败: {str(e)}")
        
        # 解析关键参数
        import re
        
        def extract_param(pattern, default=""):
            match = re.search(pattern, script_content)
            return match.group(1) if match else default
        
        config = {
            "train_batch_size": int(extract_param(r'data\.train_batch_size=(\d+)', "16")),
            "learning_rate": extract_param(r'actor\.optim\.lr=([\d.e-]+)', "1e-6"),
            "total_epochs": int(extract_param(r'trainer\.total_epochs=(\d+)', "1")),
            "n_gpus": int(extract_param(r'trainer\.n_gpus_per_node=(\d+)', "2")),
            "save_freq": int(extract_param(r'trainer\.save_freq=(\d+)', "20")),
            "max_turns": int(extract_param(r'rollout\.max_turns=(\d+)', "2")),
            "kl_coef": extract_param(r'kl_ctrl\.kl_coef=([\d.e-]+)', "0.001"),
            "gpu_memory_utilization": extract_param(r'gpu_memory_utilization=([\d.]+)', "0.6"),
        }
        
        return {
            "status": "success",
            "project_id": project_id,
            "config": config,
            "script_path": str(script_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练配置失败: {str(e)}")


@app.post("/rl/projects/{project_id}/training-config")
async def update_training_config(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新训练脚本的配置参数
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        config = payload.get('config', {})
        
        # 读取训练脚本
        script_path = project_path / f"{project_id}.sh"
        if not script_path.exists():
            raise HTTPException(status_code=404, detail=f"未找到训练脚本: {script_path}")
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取训练脚本失败: {str(e)}")
        
        # 备份原脚本
        backup_path = script_path.with_suffix('.sh.bak')
        try:
            import shutil
            shutil.copy2(script_path, backup_path)
            logger.info(f"已备份训练脚本到: {backup_path}")
        except Exception as e:
            logger.warning(f"备份脚本失败: {e}")
        
        # 更新参数
        import re
        
        def replace_param(content, pattern, new_value):
            return re.sub(pattern, rf'\g<1>{new_value}\3', content)
        
        if 'train_batch_size' in config:
            script_content = replace_param(
                script_content,
                r'(data\.train_batch_size=)(\d+)(\\)',
                config['train_batch_size']
            )
        
        if 'learning_rate' in config:
            script_content = replace_param(
                script_content,
                r'(actor\.optim\.lr=)([\d.e-]+)(\\)',
                config['learning_rate']
            )
        
        if 'total_epochs' in config:
            script_content = replace_param(
                script_content,
                r'(trainer\.total_epochs=)(\d+)( )',
                config['total_epochs']
            )
        
        if 'n_gpus' in config:
            script_content = replace_param(
                script_content,
                r'(trainer\.n_gpus_per_node=)(\d+)(\\)',
                config['n_gpus']
            )
        
        if 'save_freq' in config:
            script_content = replace_param(
                script_content,
                r'(trainer\.save_freq=)(\d+)(\\)',
                config['save_freq']
            )
        
        if 'max_turns' in config:
            script_content = replace_param(
                script_content,
                r'(rollout\.max_turns=)(\d+)(\\)',
                config['max_turns']
            )
        
        if 'kl_coef' in config:
            script_content = replace_param(
                script_content,
                r'(kl_ctrl\.kl_coef=)([\d.e-]+)(\\)',
                config['kl_coef']
            )
        
        if 'gpu_memory_utilization' in config:
            script_content = replace_param(
                script_content,
                r'(gpu_memory_utilization=)([\d.]+)(\\)',
                config['gpu_memory_utilization']
            )
        
        # 保存更新后的脚本
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存训练脚本失败: {str(e)}")
        
        logger.info(f"成功更新训练配置: {script_path}")
        
        return {
            "status": "success",
            "message": "训练配置更新成功",
            "project_id": project_id,
            "script_path": str(script_path),
            "updated_params": list(config.keys()),
            "backup_created": backup_path.exists()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新训练配置失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"更新训练配置失败: {str(e)}")


@app.post("/rl/projects/{project_id}/training/start")
async def start_training(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    启动训练脚本
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        script_path = project_path / f"{project_id}.sh"
        if not script_path.exists():
            raise HTTPException(status_code=404, detail=f"未找到训练脚本: {script_path}")
        
        # 检查脚本是否有执行权限
        if not os.access(script_path, os.X_OK):
            try:
                os.chmod(script_path, 0o755)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"无法设置脚本执行权限: {str(e)}")
        
        # 使用后台进程启动训练
        import subprocess
        
        log_file = project_path / "training.log"
        pid_file = project_path / "training.pid"
        
        # 检查是否已有训练在运行
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # 检查进程是否还在运行
                try:
                    os.kill(old_pid, 0)  # 发送信号0检查进程是否存在
                    raise HTTPException(
                        status_code=400,
                        detail=f"训练已在运行中 (PID: {old_pid})，请先停止当前训练"
                    )
                except OSError:
                    # 进程不存在，删除旧的 PID 文件
                    pid_file.unlink()
            except (ValueError, FileNotFoundError):
                pid_file.unlink()
        
        # 启动训练进程
        try:
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    ['bash', str(script_path)],
                    cwd=str(project_path),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True  # 创建新的进程组
                )
            
            # 保存 PID
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
            
            logger.info(f"训练已启动: {script_path}, PID: {process.pid}")
            
            return {
                "status": "success",
                "message": "训练已启动",
                "project_id": project_id,
                "pid": process.pid,
                "log_file": str(log_file),
                "script_path": str(script_path)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")


@app.post("/rl/projects/{project_id}/training/stop")
async def stop_training(project_id: str) -> Dict[str, Any]:
    """
    停止训练脚本
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        pid_file = project_path / "training.pid"
        
        if not pid_file.exists():
            raise HTTPException(status_code=404, detail="未找到运行中的训练进程")
        
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 尝试终止进程
            import signal
            try:
                # 先发送 SIGTERM，给进程优雅退出的机会
                os.kill(pid, signal.SIGTERM)
                
                # 等待一段时间
                import time
                time.sleep(2)
                
                # 检查进程是否还存在
                try:
                    os.kill(pid, 0)
                    # 如果还存在，强制终止
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass  # 进程已退出
                
                # 删除 PID 文件
                pid_file.unlink()
                
                logger.info(f"已停止训练进程: PID {pid}")
                
                return {
                    "status": "success",
                    "message": f"训练进程已停止 (PID: {pid})",
                    "project_id": project_id,
                    "pid": pid
                }
                
            except OSError as e:
                # 进程不存在
                pid_file.unlink()
                raise HTTPException(status_code=404, detail=f"训练进程不存在 (PID: {pid})")
            
        except ValueError:
            pid_file.unlink()
            raise HTTPException(status_code=500, detail="PID 文件格式错误")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"停止训练失败: {str(e)}")

@app.get("/rl/projects/{project_id}/training/status")
async def get_training_status(project_id: str) -> Dict[str, Any]:
    """
    获取训练状态
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        pid_file = project_path / "training.pid"
        log_file = project_path / "training.log"
        
        is_running = False
        pid = None
        
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # 检查进程是否存在且正在运行
                try:
                    os.kill(pid, 0)
                    
                    # 进一步检查进程状态
                    if hasattr(os, 'waitpid'):
                        try:
                            result = os.waitpid(pid, os.WNOHANG)
                            if result[0] == 0:
                                is_running = True
                            else:
                                # 进程已退出
                                logger.info(f"训练进程 {pid} 已退出")
                                pid_file.unlink(missing_ok=True)
                                is_running = False
                        except ChildProcessError:
                            is_running = True
                    else:
                        is_running = True
                        
                except OSError:
                    # 进程不存在
                    logger.info(f"训练进程 {pid} 不存在，清理 PID 文件")
                    pid_file.unlink(missing_ok=True)
                    is_running = False
                    
            except (ValueError, IOError):
                is_running = False
        
        # 读取日志最后几行
        log_tail = ""
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    log_tail = ''.join(lines[-50:]) if lines else ""
            except Exception as e:
                logger.error(f"读取日志文件失败: {e}")
        
        return {
            "status": "success",
            "project_id": project_id,
            "is_running": is_running,
            "pid": pid if is_running else None,
            "log_file": str(log_file) if log_file.exists() else None,
            "log_tail": log_tail
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练状态失败: {str(e)}")

# Add these new endpoints before the start_server function

@app.get("/rl/projects/{project_id}/tensorboard/metrics")
async def get_tensorboard_metrics(project_id: str) -> Dict[str, Any]:
    """
    获取项目的 TensorBoard 指标数据
    返回8个重要指标：score, reward, pg_loss, pg_clipfrac, ppo_kl, grad_norm, advantages, returns
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        # 查找 tensorboard_log 目录
        tensorboard_dir = project_path / "tensorboard_log"
        if not tensorboard_dir.exists():
            return {
                "status": "success",
                "project_id": project_id,
                "has_data": False,
                "message": "TensorBoard 日志目录不存在"
            }
        
        # 递归查找所有 events 文件
        event_files = list(tensorboard_dir.glob("**/events.out.tfevents.*"))
        if not event_files:
            return {
                "status": "success",
                "project_id": project_id,
                "has_data": False,
                "message": "没有找到 TensorBoard 事件文件"
            }
        
        # 使用最新的 event 文件目录
        latest_event_dir = max([f.parent for f in event_files], key=lambda p: p.stat().st_mtime)
        
        logger.info(f"加载 TensorBoard 数据从: {latest_event_dir}")
        
        # 加载 TensorBoard 数据，配置 size_guidance 以加载所有数据
        size_guidance = {
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,  # 0 表示加载所有标量数据
            event_accumulator.HISTOGRAMS: 0,
        }
        ea = event_accumulator.EventAccumulator(str(latest_event_dir), size_guidance=size_guidance)
        ea.Reload()
        
        # 获取所有可用的标签
        tags = ea.Tags()
        available_tags = tags.get("scalars", [])
        
        logger.info(f"可用的标量标签数量: {len(available_tags)}")
        
        # 如果没有标量数据，返回调试信息
        if not available_tags:
            return {
                "status": "success",
                "project_id": project_id,
                "has_data": False,
                "message": f"TensorBoard 日志中没有找到标量数据。目录: {latest_event_dir}"
            }
        
        # 定义要提取的8个关键指标（基于实际的标签名称）
        metric_mappings = {
            "score": ["critic/score/mean"],
            "reward": ["critic/rewards/mean"],
            "pg_loss": ["actor/pg_loss"],
            "pg_clipfrac": ["actor/pg_clipfrac"],
            "ppo_kl": ["actor/ppo_kl"],
            "grad_norm": ["actor/grad_norm"],
            "advantages": ["critic/advantages/mean"],
            "returns": ["critic/returns/mean"]
        }
        
        # 提取数据
        metrics_data = {}
        
        for metric_name, possible_tags in metric_mappings.items():
            found = False
            for tag in possible_tags:
                if tag in available_tags:
                    try:
                        events = ea.Scalars(tag)
                        if events:  # 确保有数据
                            metrics_data[metric_name] = {
                                "tag": tag,
                                "data": [{"step": e.step, "value": float(e.value), "wall_time": e.wall_time} 
                                        for e in events]
                            }
                            logger.info(f"成功读取指标 {metric_name} (标签: {tag}), 数据点数: {len(events)}")
                            found = True
                            break
                    except Exception as e:
                        logger.warning(f"读取指标 {tag} 失败: {e}")
                        continue
            
            if not found:
                metrics_data[metric_name] = {
                    "tag": None,
                    "data": []
                }
                logger.warning(f"未找到指标 {metric_name}")
        
        # 计算统计信息
        summary = {}
        for metric_name, metric_info in metrics_data.items():
            if metric_info["data"]:
                values = [d["value"] for d in metric_info["data"]]
                summary[metric_name] = {
                    "current": values[-1] if values else None,
                    "max": max(values) if values else None,
                    "min": min(values) if values else None,
                    "avg": sum(values) / len(values) if values else None,
                    "count": len(values)
                }
            else:
                summary[metric_name] = {
                    "current": None,
                    "max": None,
                    "min": None,
                    "avg": None,
                    "count": 0
                }
        
        # 统计找到的指标数量
        found_metrics = sum(1 for m in metrics_data.values() if m["data"])
        logger.info(f"总共找到 {found_metrics}/{len(metric_mappings)} 个指标")
        
        # 获取训练进度信息
        current_epoch = None
        total_epochs = None
        current_step = None
        
        # 尝试从 training/epoch 获取当前 epoch
        if "training/epoch" in available_tags:
            try:
                epoch_events = ea.Scalars("training/epoch")
                if epoch_events:
                    current_epoch = epoch_events[-1].value
            except Exception as e:
                logger.warning(f"读取 epoch 失败: {e}")
        
        # 尝试从 training/global_step 获取当前 step
        if "training/global_step" in available_tags:
            try:
                step_events = ea.Scalars("training/global_step")
                if step_events:
                    current_step = step_events[-1].value
            except Exception as e:
                logger.warning(f"读取 global_step 失败: {e}")
        
        # 尝试从训练脚本读取 total_epochs
        script_path = project_path / f"{project_id}.sh"
        if script_path.exists():
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_content = f.read()
                    import re
                    match = re.search(r'trainer\.total_epochs=(\d+)', script_content)
                    if match:
                        total_epochs = int(match.group(1))
            except Exception as e:
                logger.warning(f"读取 total_epochs 失败: {e}")
        
        # 检查训练进程状态
        pid_file = project_path / "training.pid"
        is_training = False
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                try:
                    os.kill(pid, 0)
                    is_training = True
                except OSError:
                    is_training = False
            except (ValueError, IOError):
                is_training = False
        
        return {
            "status": "success",
            "project_id": project_id,
            "has_data": True,
            "metrics": metrics_data,
            "summary": summary,
            "log_dir": str(latest_event_dir),
            "found_metrics_count": found_metrics,
            "training_status": {
                "is_running": is_training,
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "current_step": current_step,
                "progress_percent": (current_epoch / total_epochs * 100) if (current_epoch is not None and total_epochs is not None and total_epochs > 0) else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取 TensorBoard 指标失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取 TensorBoard 指标失败: {str(e)}")

@app.get("/rl/tensorboard/projects")
async def get_tensorboard_projects() -> Dict[str, Any]:
    """
    获取所有有 TensorBoard 数据的项目列表
    """
    try:
        projects_dir = Path(PROJECT_BASE_DIR)
        if not projects_dir.exists():
            return {"status": "success", "projects": []}
        
        projects = []
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            
            tensorboard_dir = project_dir / "tensorboard_log"
            if tensorboard_dir.exists():
                event_files = list(tensorboard_dir.glob("**/events.out.tfevents.*"))
                if event_files:
                    latest_time = max([f.stat().st_mtime for f in event_files])
                    projects.append({
                        "id": project_dir.name,
                        "name": project_dir.name,
                        "has_tensorboard": True,
                        "last_update": latest_time
                    })
        
        # 按最后更新时间降序排序
        projects.sort(key=lambda x: x.get("last_update", 0), reverse=True)
        
        return {
            "status": "success",
            "projects": projects
        }
        
    except Exception as e:
        logger.error(f"获取 TensorBoard 项目列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取项目列表失败: {str(e)}")

@app.get("/rl/projects/{project_id}/tensorboard/check-update")
async def check_tensorboard_update(project_id: str) -> Dict[str, Any]:
    """
    轻量级检查：返回 TensorBoard 数据的更新状态
    用于前端判断是否需要拉取完整数据
    """
    try:
        project_id = project_id.strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="项目名称不能为空")
        
        project_path = Path(PROJECT_BASE_DIR) / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"项目 '{project_id}' 不存在")
        
        # 查找 tensorboard_log 目录
        tensorboard_dir = project_path / "tensorboard_log"
        if not tensorboard_dir.exists():
            return {
                "status": "success",
                "has_data": False,
                "last_modified": 0,
                "data_signature": None
            }
        
        # 递归查找所有 events 文件
        event_files = list(tensorboard_dir.glob("**/events.out.tfevents.*"))
        if not event_files:
            return {
                "status": "success",
                "has_data": False,
                "last_modified": 0,
                "data_signature": None
            }
        
        # 获取最新文件的修改时间
        latest_event_file = max(event_files, key=lambda p: p.stat().st_mtime)
        last_modified = latest_event_file.stat().st_mtime
        
        # 检查训练状态
        pid_file = project_path / "training.pid"
        is_training = False
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                try:
                    os.kill(pid, 0)
                    is_training = True
                except OSError:
                    is_training = False
            except (ValueError, IOError):
                is_training = False
        
        # 生成数据签名（用于快速检测变化）
        data_signature = f"{last_modified}_{len(event_files)}_{is_training}"
        
        return {
            "status": "success",
            "has_data": True,
            "last_modified": last_modified,
            "data_signature": data_signature,
            "is_training": is_training,
            "event_files_count": len(event_files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检查 TensorBoard 更新失败: {e}")
        raise HTTPException(status_code=500, detail=f"检查更新失败: {str(e)}")

def start_server(host: str = "0.0.0.0", port: int = 8228):
    """
    启动 FastAPI 服务器
    
    Args:
        host: 监听地址
        port: 监听端口
    """
    logger.info(f"Starting AgentClient API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

# if __name__ == "__main__":
#     agent_client = AgentClient(STORE_URL, project_name="my_weather_query")
#     agent_client.download_agent("my_weather_query")
#     agent_client.build_parquet("report_writer")
#     agent_client.register_env_to_registry("my_weather_query")
#     agent_client.generate_training_script("my_weather_query")
if __name__ == "__main__":
    # 作为服务器运行
    start_server()