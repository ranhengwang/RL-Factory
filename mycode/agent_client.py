import requests
import json
import sys
import os
import re  # 新增：用于正则匹配修改 __init__.py

# 导入代码生成模块
try:
    from mycode.json_2_tool import generate_mcp_tool_code, find_function_def
except ImportError:
    try:
        from json_2_tool import generate_mcp_tool_code, find_function_def
    except ImportError:
        print("❌ 无法导入 json_2_tool 模块，请确保它在当前目录或 PYTHONPATH 中。")
        sys.exit(1)

# 服务器基础地址
BASE_URL = "http://127.0.0.1:8000"

# 获取当前脚本所在目录的绝对路径 (mycode)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_BASE_DIR = os.path.join(CURRENT_DIR, "project")

def fetch_agent_config():
    """步骤 1: 获取 Agent 基础配置"""
    url = f"{BASE_URL}/get_agent_config"
    print(f"\n[1/3] 正在获取 Agent 配置: {url} ...")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        config = response.json()
        print(f"✅ 获取成功: Agent Name = {config.get('name')}")
        return config
    except Exception as e:
        print(f"❌ 获取 Agent 配置失败: {e}")
        sys.exit(1)

def create_project_structure(agent_name):
    """创建项目目录结构"""
    project_dir = os.path.join(PROJECT_BASE_DIR, agent_name)
    
    dirs_to_create = [
        os.path.join(project_dir, "data"),
        os.path.join(project_dir, "envs"),
        os.path.join(project_dir, "output"),
        os.path.join(project_dir, "envs", "tools"),
        os.path.join(project_dir, "envs", "configs"),
        os.path.join(project_dir, "envs", "reward"),
    ]
    
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"    📁 创建目录: {d}")
        
    return project_dir

def fetch_model_path(model_name):
    """步骤 2: 获取模型存储路径"""
    url = f"{BASE_URL}/get_model_path"
    print(f"\n[2/3] 正在查询模型路径: {model_name} ...")
    try:
        # 使用 params 传递查询参数
        response = requests.get(url, params={"model_name": model_name}, timeout=5)
        response.raise_for_status()
        data = response.json()
        path = data.get("path")
        print(f"✅ 模型路径: {path}")
        return path
    except Exception as e:
        print(f"❌ 获取模型路径失败: {e}")
        return None

def fetch_tool_config(tool_name):
    """步骤 3: 获取单个工具的配置"""
    url = f"{BASE_URL}/get_tool_config"
    try:
        response = requests.get(url, params={"tool_name": tool_name}, timeout=5)
        response.raise_for_status()
        config = response.json()
        return config
    except Exception as e:
        print(f"         ❌ 获取工具 {tool_name} 失败: {e}")
        return None

def save_tool_code(project_dir, tool_name, code):
    """保存生成的工具代码到文件"""
    output_dir = os.path.join(project_dir, "envs", "tools")
    output_path = os.path.join(output_dir, f"{tool_name}.py")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"    💾 代码已保存: {output_path}")
        return output_path # 返回绝对路径
    except Exception as e:
        print(f"    ❌ 保存代码失败: {e}")
        return None

def generate_mcp_config(project_dir, agent_name, tool_paths):
    """
    生成 MCP 配置文件 (.pydata)
    """
    config_dir = os.path.join(project_dir, "envs", "configs")
    
    # 构建配置字典
    mcp_servers = {}
    for tool_name, abs_path in tool_paths.items():
        mcp_servers[tool_name] = {
            'command': 'python3',
            'args': [abs_path] # 使用绝对路径
        }
    
    config_data = [
        {'mcpServers': mcp_servers}
    ]
    
    config_content = json.dumps(config_data, ensure_ascii=False, indent=4)

    file_name = f"{agent_name}.pydata"
    file_path = os.path.join(config_dir, file_name)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"\n✅ MCP 配置文件已生成: {file_path}")
        return file_path
    except Exception as e:
        print(f"\n❌ 生成配置文件失败: {e}")
        return None

def register_env_to_registry(agent_name):
    """
    自动注册环境到 envs/__init__.py
    """
    # 1. 确定路径和名称
    # 假设 agent_client.py 在 mycode/ 下，envs/ 在 mycode/../envs/
    root_dir = os.path.dirname(CURRENT_DIR) # RL-Factory root
    init_file_path = os.path.join(root_dir, "envs", "__init__.py")
    
    # 转换为类名 (snake_case -> CamelCase)
    # e.g. travel -> TravelEnv, travel_planner -> TravelPlannerEnv
    class_name = "".join([word.capitalize() for word in agent_name.split('_')]) + "Env"
    
    # 导入路径: mycode.project.{agent_name}.envs.reward.{agent_name}
    import_path = f"mycode.project.{agent_name}.envs.reward.{agent_name}"
    import_stmt = f"from {import_path} import {class_name}"
    
    if not os.path.exists(init_file_path):
        print(f"❌ 找不到 {init_file_path}，跳过注册。")
        return

    print(f"\n[Auto-Register] 正在注册环境到 {init_file_path} ...")
    
    try:
        with open(init_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if import_stmt in content:
            print(f"✅ 环境 {agent_name} 已注册 (Import exists)。")
            return

        # 2. 修改文件内容
        # 2.1 添加 import (在 __all__ 之前插入)
        if "__all__" in content:
            content = content.replace("__all__", f"{import_stmt}\n\n__all__", 1)
        else:
            content = f"{import_stmt}\n{content}"
            
        # 2.2 更新 __all__ 列表
        # 查找 __all__ = [ ... ]
        match_all = re.search(r"(__all__\s*=\s*\[)(.*?)(\])", content, re.DOTALL)
        if match_all:
            current_list = match_all.group(2)
            if class_name not in current_list:
                # 追加到列表末尾
                new_list = f"{current_list}, '{class_name}'"
                content = content.replace(match_all.group(0), f"{match_all.group(1)}{new_list}{match_all.group(3)}")
                
        # 2.3 更新 TOOL_ENV_REGISTRY 字典
        # 查找 TOOL_ENV_REGISTRY = { ... }
        match_reg = re.search(r"(TOOL_ENV_REGISTRY\s*=\s*\{)(.*?)(\})", content, re.DOTALL)
        if match_reg:
            current_dict = match_reg.group(2)
            # 检查 key 是否存在
            if f"'{agent_name}'" not in current_dict and f'"{agent_name}"' not in current_dict:
                new_entry = f",\n    '{agent_name}': {class_name}"
                content = content.replace(match_reg.group(0), f"{match_reg.group(1)}{current_dict}{new_entry}{match_reg.group(3)}")

        # 3. 写回文件
        with open(init_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 注册成功: {class_name} -> {init_file_path}")
        print(f"   Import: {import_stmt}")
        
    except Exception as e:
        print(f"❌ 注册失败: {e}")

def generate_training_script(project_dir, agent_name, model_path):
    """
    生成训练脚本 (.sh)
    """
    script_path = os.path.join(project_dir, f"{agent_name}.sh")
    
    # 构造关键路径
    train_file = os.path.join(project_dir, "data", f"{agent_name}_train.parquet")
    val_file = train_file # 假设验证集同训练集
    config_path = os.path.join(project_dir, "envs", "configs", f"{agent_name}.pydata")
    result_dir = os.path.join(project_dir, "output")
    
    # 模板内容 (基于 travel.sh)
    script_content = f"""#!/bin/bash
# GRPO Training Script for {agent_name}
# Auto-generated by agent_client.py

set -e -x
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_COMPILE_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export VLLM_TORCH_COMPILE_DISABLE=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu

export RAY_RUNTIME_ENV_IGNORE_WEBHOOK=1
unset ROCR_VISIBLE_DEVICES

# Model Paths
export MODEL_PATH=${{MODEL_PATH:-{model_path}}}
export REWARD_MODEL_PATH=/home/ranhengwang/.cache/modelscope/hub/models/Qwen/Qwen3-8B
export RESULT_DIR=${{RESULT_DIR:-{result_dir}}}

python3 -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \\
    algorithm.adv_estimator=grpo\\
    data.train_files={train_file}\\
    data.val_files={val_file}\\
    data.train_batch_size=16\\
    data.max_prompt_length=4096\\
    data.max_response_length=512\\
    actor_rollout_ref.model.path=$MODEL_PATH\\
    actor_rollout_ref.model.use_remove_padding=True\\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\\
    actor_rollout_ref.actor.optim.lr=1e-6\\
    actor_rollout_ref.actor.ppo_mini_batch_size=8\\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2\\
    actor_rollout_ref.actor.use_kl_loss=True\\
    actor_rollout_ref.actor.kl_loss_coef=0.001\\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl\\
    actor_rollout_ref.actor.fsdp_config.param_offload=True\\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\\
    actor_rollout_ref.actor.state_masking=True\\
    actor_rollout_ref.actor.use_torch_compile=False\\
    actor_rollout_ref.ref.use_torch_compile=False\\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4\\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\\
    actor_rollout_ref.rollout.name=vllm\\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6\\
    actor_rollout_ref.rollout.n=4\\
    actor_rollout_ref.rollout.max_turns=2\\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4\\
    actor_rollout_ref.ref.fsdp_config.param_offload=True\\
    actor_rollout_ref.rollout.enforce_eager=False\\
    actor_rollout_ref.rollout.free_cache_engine=True\\
    actor_rollout_ref.env.name={agent_name}\\
    actor_rollout_ref.env.mcp_mode=stdio\\
    actor_rollout_ref.env.tool_manager=qwen3\\
    actor_rollout_ref.env.enable_thinking=False\\
    actor_rollout_ref.env.config_path={config_path}\\
    actor_rollout_ref.env.use_process_reward=True\\
    reward_rollout.if_use_reward_rollout=False\\
    reward_rollout.rollout.tensor_model_parallel_size=4\\
    reward_rollout.rollout.gpu_memory_utilization=0.6\\
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\\
    reward_rollout.rollout.free_cache_engine=True\\
    reward_rollout.rollout.response_length=2048\\
    reward_model.reward_manager=parallel\\
    algorithm.kl_ctrl.kl_coef=0.001\\
    trainer.critic_warmup=0\\
    trainer.logger=['tensorboard']\\
    trainer.project_name='GRPO_{agent_name}'\\
    trainer.experiment_name='{agent_name}_with_thinking'\\
    trainer.n_gpus_per_node=2\\
    trainer.nnodes=1\\
    trainer.val_before_train=False\\
    trainer.default_local_dir=$RESULT_DIR\\
    trainer.default_hdfs_dir=null\\
    trainer.save_freq=20\\
    trainer.test_freq=10\\
    trainer.total_epochs=1 $@ 2>&1 | tee grpo.log
"""
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"\n✅ 训练脚本已生成: {script_path}")
        return script_path
    except Exception as e:
        print(f"\n❌ 生成训练脚本失败: {e}")
        return None

def main():
    # 1. 获取 Agent 配置
    agent_config = fetch_agent_config()
    raw_name = agent_config.get('name', 'unknown_agent')
    
    # 清洗名称：将连字符替换为下划线，确保 Python 兼容性
    agent_name = raw_name.replace('-', '_')
    if raw_name != agent_name:
        print(f"⚠️ 将 Agent 名称 '{raw_name}' 转换为 Python 友好的 '{agent_name}'")

    print("-" * 40)
    print("🎯 Agent 提示词:")
    print(f"  - {agent_config.get('instruction')}")
    
    # 1.1 创建项目目录结构
    print(f"\n[Init] 初始化项目目录结构: {agent_name} ...")
    project_dir = create_project_structure(agent_name)

    # 2. 获取模型路径
    model_name = agent_config.get("model")
    if model_name:
        model_path = fetch_model_path(model_name)
    else:
        print("⚠️ Agent 配置中未找到 'model' 字段")

    # 3. 获取所有工具的配置
    tools = agent_config.get("system_tools", []) + agent_config.get("mcp", [])
    print(f"\n[3/3] 正在获取 {len(tools)} 个工具的详细配置并生成代码 ...")
    
    tool_configs = {}
    generated_tool_paths = {} # 记录成功生成的工具路径 {name: path}
    
    for tool_name in tools:
        config = fetch_tool_config(tool_name)
        if config:
            tool_configs[tool_name] = config
            
            # 检查是否是自动生成的 Mock 数据
            is_mock = config.get("is_mock", False)
            status_str = "⚠️ AUTO-MOCK" if is_mock else "✅ REAL"
            
            print(f"  - {tool_name}: {status_str}")
            
            # --- 生成代码逻辑 ---
            try:
                # 生成代码字符串
                code = generate_mcp_tool_code(config)
                
                # 再次确认工具名称
                func_def = find_function_def(config)
                actual_tool_name = func_def.get("name", tool_name) if func_def else tool_name
                
                # 保存文件
                abs_path = save_tool_code(project_dir, actual_tool_name, code)
                if abs_path:
                    generated_tool_paths[actual_tool_name] = abs_path
                    
            except Exception as e:
                print(f"    ❌ 代码生成失败: {e}")

            print("-" * 40)
        else:
            print(f"  - {tool_name}: FAILED")

    # 4. 生成 MCP 配置文件
    config_path = None
    if generated_tool_paths:
        config_path = generate_mcp_config(project_dir, agent_name, generated_tool_paths)
    else:
        print("\n⚠️ 没有生成任何工具，跳过配置文件生成。")

    # 5. 自动注册环境
    register_env_to_registry(agent_name)

    # 6. 生成训练脚本
    if model_path:
        generate_training_script(project_dir, agent_name, model_path)
    else:
        print("⚠️ 未找到模型路径，跳过训练脚本生成。")
        
    print("\n" + "="*40)
    print("🎉 所有信息获取完成 Summary:")
    print(f"Agent: {agent_name}")
    print(f"Project Dir: {project_dir}")
    print(f"Model Path: {model_path}")
    print(f"Tools Fetched: {len(tool_configs)}/{len(tools)}")
    print(f"Code Generated: {len(generated_tool_paths)}/{len(tools)}")
    if config_path:
        print(f"Config File: {config_path}")
    print("="*40)
    print(f"⚠️ 请记得在 {os.path.join(project_dir, 'envs', 'reward', f'{agent_name}.py')} 中添加你的奖励函数代码！")
    print(f"   类名必须为: {''.join([word.capitalize() for word in agent_name.split('_')])}Env")

if __name__ == "__main__":
    main()