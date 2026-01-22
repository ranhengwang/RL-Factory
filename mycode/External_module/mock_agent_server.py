import uvicorn
from fastapi import FastAPI
from typing import List, Dict, Any

app = FastAPI()

# 1. 模拟的 Agent 配置
# MOCK_AGENT_CONFIG = {
#   "name": "travel-planner",
#   "card": "专业的旅行规划助手，擅长协调天气、交通和住宿信息。",
#   "model": "Qwen3-4B", # 假设使用这个模型
#   "instruction": "你是一个旅行规划助手。你的任务是根据用户的需求规划行程。通常你需要先查询目的地的天气，然后查询合适的航班，最后预订酒店。请一步步使用工具完成任务。",
#   "max_actions": 10,
#   "mcp": [],
#   "system_tools": [
#     "get_weather",
#     "search_flights",
#     "book_hotel"
#   ],
#   "category": "planner",
#   "tags": ["旅行", "规划", "多工具"]
# }

MOCK_AGENT_CONFIG = {
    "name": 'weather_query',
    "card": '专业的天气查询助手，能够查询全球各地的实时天气信息，包括温度、湿度、风力、降水等详细数据，提供友好的天气服务和出行建议',
    "model": 'deepseek-v3.2-exp',
    "instruction": '你是一个专业的天气查询助手。你的主要职责是帮助用户查询全球各地的天气信息。\n\n## 核心能力\n- 使用queryCityWeather MCP服务查询实时天气数据\n- 提供温度、湿度、风力、降水概率等详细信息\n- 支持查询今日天气和未来几天的天气预报\n- 根据天气情况提供适当的出行建议\n\n## 交互规范\n1. 当用户询问天气时，主动询问具体城市或地区名称\n2. 使用友好的语气与用户交流\n3. 提供清晰、准确的天气信息展示\n4. 根据天气状况给出实用的建议（如带伞、穿衣建议等）\n5. 如果查询失败，友好地提示用户并提供备选方案\n\n## 注意事项\n- 确保查询的城市名称准确\n- 如果用户提供的位置不明确，主动询问具体城市\n- 保持回复的专业性和友好性',
    "max_actions": 20,
    "mcp": [
        'FastMCP-API-Server'
    ],
    "system_tools": [],
    "category": 'utilities',
    "tags": [
        'weather',
        'query',
        'forecast',
        'travel',
        'daily'
    ]
}


# 2. 模拟的模型路径
MOCK_MODEL_PATHS = {
    "claude-sonnet-4-5": "/data/models/weights/claude-sonnet-4-5-v1",
    "qwen-2.5-72b": "/data/models/weights/qwen-2.5-72b-instruct",
    "default": "/data/models/weights/default-model",
    "Qwen3-4B": "/home/ranhengwang/.cache/modelscope/hub/models/Qwen/Qwen3-4B"
}

# 3. 模拟的工具配置列表 (改为列表，每个元素严格遵守您的格式)
MOCK_TOOL_CONFIGS = [
    {
        "function": {
            "name": "export_graph_to_document",
            "description": "将当前图导出为文档。",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string", "description": "图的ID"},
                    "format": {"type": "string", "description": "导出格式", "enum": ["json", "yaml"]}
                },
                "required": ["graph_id"]
            }
        }
    },
    {
        "function": {
            "name": "get_graph_spec",
            "description": "获取工作流规范文档，包含完整的设计指南。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
        {
        "function": {
            "name": "get_weather",
            "description": "查询指定城市在特定日期的天气情况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "function": {
            "name": "search_flights",
            "description": "查询从出发地到目的地的航班信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "出发城市"},
                    "destination": {"type": "string", "description": "抵达城市"},
                    "date": {"type": "string", "description": "出发日期"}
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "function": {
            "name": "book_hotel",
            "description": "预订指定城市的酒店。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "check_in": {"type": "string", "description": "入住日期"},
                    "nights": {"type": "integer", "description": "入住晚数"}
                },
                "required": ["city"]
            }
        }
    }
]

MOCK_TRAJECTORY = {
    "status": "success",
    "taskType": "report_writer",
    "description": "这是一个报告写作任务,……",
    "count": 16,
     "data": [
        {
            "id": 48,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "提高计算模块的代码覆盖率",
                    "action": "Write Unit Tests",
                    "observation": "新增 5 个边界值测试用例",
                    "reward": 0.7
                },
                {
                    "stepID": 2,
                    "thought": "运行测试套件",
                    "action": "Execute Pytest",
                    "observation": "覆盖率提升至 95%",
                    "reward": 1.0
                }
            ],
            "length": 2,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 47,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "根据长篇文章总结核心观点",
                    "action": "Summarize Content",
                    "observation": "提取出3条核心结论",
                    "reward": 0.8
                }
            ],
            "length": 1,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 46,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "无法连接远程数据库",
                    "action": "Ping Host",
                    "observation": "网络不通，请求超时",
                    "reward": 0.2
                },
                {
                    "stepID": 2,
                    "thought": "检查安全组规则",
                    "action": "Modify Security Group",
                    "observation": "3306 端口未对内网开放，已开启",
                    "reward": 1.0
                },
            ],
            "length": 2,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 45,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "按钮在移动端显示不全",
                    "action": "Update CSS",
                    "observation": "响应式布局已修复",
                    "reward": 0.9
                }
            ],
            "length": 1,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 44,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "分析竞品功能点",
                    "action": "Browse Competitor Web",
                    "observation": "整理出5个核心差异点",
                    "reward": 0.7
                }
            ],
            "length": 1,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 43,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "对新接口进行压力测试",
                    "action": "Run JMeter",
                    "observation": "QPS 达到 1000，响应时间稳定",
                    "reward": 0.8
                }
            ],
            "length": 1,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 42,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "整理项目文件夹",
                    "action": "Move Files",
                    "observation": "旧文档已移至 Archive 目录",
                    "reward": 0.4
                }
            ],
            "length": 1,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 41,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "客户询问产品报价单",
                    "action": "Draft Email",
                    "observation": "邮件草稿已写好，包含最新价目表附件",
                    "reward": 0.8
                }
            ],
            "length": 1,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 40,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "将这段英文技术文档翻译成中文",
                    "action": "Translate text",
                    "observation": "翻译完成，但 'Latency' 一词需校对",
                    "reward": 0.7
                },
                {
                    "stepID": 2,
                    "thought": "根据语境将延迟统一为 '时延'",
                    "action": "Refine Translation",
                    "observation": "文本通顺，符合技术规范",
                    "reward": 0.9
                }
            ],
            "length": 2,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 39,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "Excel 表格中存在大量重复项",
                    "action": "Remove Duplicates",
                    "observation": "删除了 45 条重复记录",
                    "reward": 0.6
                }
            ],
            "length": 1,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 38,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "协调三个人的时间开周会",
                    "action": "Check Calendar",
                    "observation": "周二下午2点三方都有空",
                    "reward": 0.5
                },
                {
                    "stepID": 2,
                    "thought": "发送会议邀约",
                    "action": "Send Invite",
                    "observation": "所有人已接受邀请",
                    "reward": 1.0
                }
            ],
            "length": 2,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 37,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "查询运行缓慢，检查执行计划",
                    "action": "EXPLAIN ANALYZE",
                    "observation": "发现全表扫描，主键索引未生效",
                    "reward": 0.4
                },
                {
                    "stepID": 2,
                    "thought": "尝试强制指定索引",
                    "action": "Add Index Hint",
                    "observation": "耗时从 2s 降低至 50ms",
                    "reward": 0.9
                }
            ],
            "length": 2,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 36,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "搜索关于大语言模型长文本处理的论文",
                    "action": "Search Google Scholar",
                    "observation": "获得10篇相关文献，涉及RAG和FlashAttention",
                    "reward": 0.7
                }
            ],
            "length": 1,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 35,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "用户反馈登录报错，查看后端日志",
                    "action": "Read Logs",
                    "observation": "发现 NullPointerException 在 AuthService 24行",
                    "reward": 0.6
                },
                {
                    "stepID": 2,
                    "thought": "分析是由于空用户对象引起的，添加空值判断",
                    "action": "Fix Code",
                    "observation": "测试用例通过",
                    "reward": 1.0
                }
            ],
            "length": 2,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 34,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "检查引用格式",
                    "action": "Check References",
                    "observation": "发现2处格式错误",
                    "reward": 0.3
                }
            ],
            "length": 1,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        },
        {
            "id": 33,
            "steps": [
                {
                    "stepID": 1,
                    "thought": "收到写报告的任务，首先需要列大纲",
                    "action": "Create Outline",
                    "observation": "大纲已生成，包含3个主要部分",
                    "reward": 0.5
                },
                {
                    "stepID": 2,
                    "thought": "开始撰写第一部分：背景介绍",
                    "action": "Write Introduction",
                    "observation": "第一段落完成，字数200字",
                    "reward": 0.8
                }
            ],
            "length": 2,
            "evaluated": 1,
            "created_at": "2025-12-29 09:33:38",
            "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
            "golden_answers": ["已完成機票與酒店預訂"]
        }
    ]
}

@app.get("/get_agent_config")
def get_agent_config():
    return MOCK_AGENT_CONFIG

@app.get("/get_model_path")
def get_model_path(model_name: str):
    path = MOCK_MODEL_PATHS.get(model_name)
    if not path:
        return {"model_name": model_name, "path": MOCK_MODEL_PATHS["default"], "note": "Using default path"}
    return {"model_name": model_name, "path": path}

@app.get("/get_tool_config")
def get_tool_config(tool_name: str):
    """根据工具名称返回工具配置 JSON"""
    
    # 遍历列表查找匹配的工具
    found_config = None
    for tool_item in MOCK_TOOL_CONFIGS:
        # 检查是否存在 function 键，且 name 匹配
        if "function" in tool_item and tool_item["function"].get("name") == tool_name:
            found_config = tool_item
            break
    
    # 如果找到了，直接返回
    if found_config:
        return found_config
    
    return False

@app.get("/trajectory/getTrajectory")
def get_trajectory(taskType: str = None, batchSize: int = None):
    """
    获取轨迹数据
    
    Args:
        taskType: 任务类型（可选，用于过滤）
        batchSize: 批次大小（可选，用于限制返回数量）
    
    Returns:
        Dict: 轨迹数据，格式与 MOCK_TRAJECTORY 相同
    """
    result = MOCK_TRAJECTORY.copy()
    
    # 如果指定了 taskType，可以过滤数据（当前 MOCK_TRAJECTORY 中 taskType 是 "report_writer"）
    if taskType and result.get("taskType") != taskType:
        # 如果 taskType 不匹配，返回空数据或保持原样
        # 这里可以根据需要调整逻辑
        pass
    
    # 如果指定了 batchSize，限制返回的数据数量
    if batchSize and batchSize > 0:
        data_list = result.get("data", [])
        if len(data_list) > batchSize:
            result["data"] = data_list[:batchSize]
            result["count"] = len(result["data"])
    
    return result

if __name__ == "__main__":
    print("启动模拟 Agent 配置服务器...")
    print("访问地址: http://127.0.0.1:8000/get_agent_config")
    print("轨迹数据接口: http://127.0.0.1:8000/trajectory/getTrajectory")
    uvicorn.run(app, host="127.0.0.1", port=8000)