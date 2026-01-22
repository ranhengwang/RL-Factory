import json
from typing import Any, Optional, List, Dict
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务
mcp = FastMCP("LocalServer")

@mcp.tool()
def book_hotel(city: str, check_in: str = None, nights: int = None):
    """
    预订指定城市的酒店。
    
    Args:
        city: 城市名称
        check_in: 入住日期
        nights: 入住晚数
        
    Returns:
        dict: 执行结果
    """
    try:
        # TODO: 在此处实现真实的业务逻辑
        # 当前为自动生成的 Mock 实现，仅打印参数并返回成功状态
        
        print(f"Tool 'book_hotel' called.")
        
        # 模拟处理过程
        result_data = {
            "status": "success",
            "message": f"Successfully executed tool: book_hotel",
            "input_parameters": {
                'city': city,
                'check_in': check_in,
                'nights': nights,
            }
        }
        
        return result_data

    except Exception as e:
       return f"⚠️ Error executing book_hotel: {str(e)}"

if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport='stdio')
