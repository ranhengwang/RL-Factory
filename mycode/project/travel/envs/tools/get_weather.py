import json
from typing import Any, Optional, List, Dict
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务
mcp = FastMCP("LocalServer")

@mcp.tool()
def get_weather(city: str, date: str = None):
    """
    查询指定城市在特定日期的天气情况。
    
    Args:
        city: 城市名称
        date: 日期，格式 YYYY-MM-DD
        
    Returns:
        dict: 执行结果
    """
    try:
        # TODO: 在此处实现真实的业务逻辑
        # 当前为自动生成的 Mock 实现，仅打印参数并返回成功状态
        
        print(f"Tool 'get_weather' called.")
        
        # 模拟处理过程
        result_data = {
            "status": "success",
            "message": f"Successfully executed tool: get_weather",
            "input_parameters": {
                'city': city,
                'date': date,
            }
        }
        
        return result_data

    except Exception as e:
       return f"⚠️ Error executing get_weather: {str(e)}"

if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport='stdio')
