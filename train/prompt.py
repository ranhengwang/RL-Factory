def get_reward_generation_prompt(agent_config: dict, tool_code: str, ref_reward_code: str) -> str:
    """
    生成用于大模型生成奖励函数的提示词
    
    Args:
        agent_config: agent 配置 JSON
        tool_code: MCP 工具代码
        ref_reward_code: 参考奖励函数代码
    
    Returns:
        完整的提示词字符串
    """
    
    agent_name = agent_config.get("agent_config", {}).get("name", "Agent")
    agent_card = agent_config.get("agent_config", {}).get("card", "")
    instruction = agent_config.get("agent_config", {}).get("instruction", "")
    mcp_list = agent_config.get("agent_config", {}).get("mcp", [])
    
    prompt = f"""# 任务：为 Agent 生成强化学习奖励函数

你是一个专业的强化学习工程师，需要根据 Agent 的配置和工具定义，生成一个完整的奖励函数类。

## Agent 信息

**Agent 名称**: {agent_name}

**Agent 描述**: {agent_card}

**Agent 指令**:
{instruction}

**可用 MCP 服务**: {', '.join(mcp_list)}

## MCP 工具代码

以下是该 Agent 可以使用的所有工具的完整定义：

```python
{tool_code}
参考奖励函数实现
以下是一个参考的奖励函数实现，请参考其结构和逻辑：
{ref_reward_code}
生成要求
请根据上述信息，为该 Agent 生成一个完整的奖励函数类，要求：

1. 类结构要求
类名必须为: {agent_name.replace('_', ' ').title().replace(' ', '')}Env（首字母大写的驼峰命名）
必须继承自 Env 基类
必须实现 __init__, set_current_trajectories, get_step_reward, _compute_score_with_rules 等核心方法
2. 工具定义要求
从 MCP 工具代码中提取所有工具的名称和必需参数
在 self.tool_schemas 中定义每个工具及其必需参数（可选参数不包含）
例如: "tool_name": ["required_param1", "required_param2"]
如果工具没有必需参数，使用空列表: "tool_name": []
3. 别名映射要求
根据工具的功能和常见叫法，在 self.action_tool_alias 中定义中英文别名
包含工具名的各种变体、常见缩写、中文翻译等
例如对于 weather 工具: "天气", "气温", "weather", "forecast" 等
4. 轨迹对齐逻辑
保持参考代码中的轨迹对齐逻辑（_traj_alignment_score 方法）
工具名归一化、文本相似度计算等辅助方法
traj_alpha 默认为 0.3
5. Step Reward 计算
格式奖励：检查工具调用格式是否正确
工具奖励：检查是否调用了目标工具
参数完整性：检查必需参数是否都提供
轨迹对齐奖励：与专家轨迹对比
6. Final Reward 计算
检查 <answer> 标签是否正确闭合
检查 <tool_call> 标签是否正确闭合
提取最终答案并给予奖励
7. 代码风格要求
添加详细的中文注释
保持代码结构清晰
使用 logger 记录调试信息（采样打印，避免刷屏）
输出格式
请直接输出完整的 Python 代码，不要包含任何解释文字。代码应该：

导入必要的模块（re, json, torch, random, logging, os, envs.base）
定义完整的 Env 类
包含所有必要的方法实现
代码可以直接保存为 .py 文件使用
现在请生成完整的奖励函数代码：
"""
    return prompt

def get_system_prompt() -> str:
    """
    获取系统提示词
    """
    return """你是一个专业的强化学习工程师和 Python 开发专家。你的任务是根据给定的 Agent 配置和工具定义，生成高质量的奖励函数代码。

你需要：

仔细分析 Agent 的功能和可用工具
从工具代码中准确提取工具签名和参数信息
设计合理的奖励机制，鼓励模型正确使用工具
生成结构清晰、注释完善的 Python 代码
确保代码可以直接运行，无需修改
请严格按照要求生成代码，不要添加任何额外的解释文字。"""