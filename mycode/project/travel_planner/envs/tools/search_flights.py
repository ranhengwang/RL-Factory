import json
from typing import Any, Optional, List, Dict
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务
mcp = FastMCP("LocalServer")

@mcp.tool()
def search_flights(origin: str, destination: str, date: str = None):
    """
    查询从出发地到目的地的航班信息。
    
    Args:
        origin: 出发城市
        destination: 抵达城市
        date: 出发日期
        
    Returns:
        dict: 执行结果
    """
    try:
        # TODO: 在此处实现真实的业务逻辑
        # 当前为自动生成的 Mock 实现，仅打印参数并返回成功状态
        
        print(f"Tool 'search_flights' called.")
        
        # 模拟处理过程
        result_data = {
            "status": "success",
            "message": f"Successfully executed tool: search_flights",
            "input_parameters": {
                'origin': origin,
                'destination': destination,
                'date': date,
            }
        }
        
        return result_data

    except Exception as e:
       return f"⚠️ Error executing search_flights: {str(e)}"

if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport='stdio')
