"""
FastMCP server with dynamic port and proper tool definition (supports arguments).
"""

import argparse
import socket
import json
import sys
import uuid
from datetime import datetime
import os
import platform
from typing import Optional
from pydantic import BaseModel

from mcp.server.fastmcp import FastMCP

# ---------------- 读取 mcp.json ----------------
with open("mcp.json", "r", encoding="utf-8") as f:
    TOOL_CONFIG = json.load(f)

# ---------------- 命令行参数 ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="Port to run the MCP server on (optional)")
args = parser.parse_args()


# ---------------- 端口检测与选择 ----------------
def is_port_free(port: int, host: str = "") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


# ---------------- 工具定义 ----------------

# 示例工具参数模型
class WeatherArgs(BaseModel):
    city: str
    unit: str = "metric"


# 数学计算工具参数模型
class MathCalcArgs(BaseModel):
    expression: str
    precision: Optional[int] = 2

    # 文件操作工具参数模型


class FileOperationArgs(BaseModel):
    operation: str  # 'read', 'write', 'list', 'size'
    path: str
    content: Optional[str] = ""

    # 系统信息工具参数模型


class SystemInfoArgs(BaseModel):
    info_type: str  # 'os', 'cpu', 'memory', 'disk', 'network'

    # 网络请求工具参数模型


class HttpRequestArgs(BaseModel):
    method: str  # 'GET', 'POST', 'PUT', 'DELETE'
    url: str
    headers: Optional[dict] = {}
    data: Optional[dict] = {}


def register_tools(api_mcp: FastMCP):
    """Register MCP tools by decorating FastMCP instance"""

    @api_mcp.tool(
        name="api.status",
        description="Check whether API server is running"
    )
    def api_status() -> str:
        return "API is running"

    @api_mcp.tool(
        name="math.calculate",
        description="Perform mathematical calculations with expression evaluation"
    )
    def math_calculate(args: MathCalcArgs):
        try:
            # 安全计算表达式（仅支持基本数学运算）
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in args.expression):
                return {"error": "Invalid characters in expression"}

            result = eval(args.expression)  # 注意：生产环境应使用更安全的方法
            if isinstance(result, float):
                result = round(result, args.precision)
            return {
                "expression": args.expression,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Calculation failed: {str(e)}"}

    @api_mcp.tool(
        name="file.operation",
        description="Perform file operations like read, write, list directory"
    )
    def file_operation(args: FileOperationArgs):
        try:
            if args.operation == "read":
                with open(args.path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {
                    "path": args.path,
                    "operation": "read",
                    "content": content,
                    "size": len(content)
                }
            elif args.operation == "write":
                with open(args.path, 'w', encoding='utf-8') as f:
                    f.write(args.content)
                return {
                    "path": args.path,
                    "operation": "write",
                    "status": "success",
                    "bytes_written": len(args.content)
                }
            elif args.operation == "list":
                files = os.listdir(args.path)
                return {
                    "path": args.path,
                    "operation": "list",
                    "files": files,
                    "count": len(files)
                }
            elif args.operation == "size":
                size = os.path.getsize(args.path)
                return {
                    "path": args.path,
                    "operation": "size",
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2)
                }
            else:
                return {"error": f"Unsupported operation: {args.operation}"}
        except Exception as e:
            return {"error": f"File operation failed: {str(e)}"}

    @api_mcp.tool(
        name="system.info",
        description="Get system information like OS, CPU, memory, disk usage"
    )
    def system_info(args: SystemInfoArgs):
        try:
            if args.info_type == "os":
                return {
                    "os": platform.system(),
                    "version": platform.version(),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "processor": platform.processor()
                }
            elif args.info_type == "cpu":
                try:
                    import psutil
                    cpu_percent = psutil.cpu_percent(interval=1)
                    cpu_count = psutil.cpu_count()
                    return {
                        "cpu_percent": cpu_percent,
                        "cpu_count": cpu_count,
                        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                    }
                except ImportError:
                    return {"error": "psutil library required for CPU info"}
            elif args.info_type == "memory":
                try:
                    import psutil
                    memory = psutil.virtual_memory()
                    return {
                        "total": f"{memory.total / (1024 ** 3):.2f} GB",
                        "available": f"{memory.available / (1024 ** 3):.2f} GB",
                        "used": f"{memory.used / (1024 ** 3):.2f} GB",
                        "percentage": memory.percent
                    }
                except ImportError:
                    return {"error": "psutil library required for memory info"}
            elif args.info_type == "disk":
                import shutil
                total, used, free = shutil.disk_usage("/")
                return {
                    "total_gb": round(total / (1024 ** 3), 2),
                    "used_gb": round(used / (1024 ** 3), 2),
                    "free_gb": round(free / (1024 ** 3), 2),
                    "usage_percent": round((used / total) * 100, 2)
                }
            elif args.info_type == "network":
                hostname = platform.node()
                import socket
                ip_address = socket.gethostbyname(hostname)
                return {
                    "hostname": hostname,
                    "ip_address": ip_address,
                    "mac_address": ':'.join(
                        ['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 2 * 6, 2)])
                }
            else:
                return {"error": f"Unsupported info type: {args.info_type}"}
        except Exception as e:
            return {"error": f"System info retrieval failed: {str(e)}"}

    @api_mcp.tool(
        name="http.request",
        description="Make HTTP requests to external services"
    )
    def http_request(args: HttpRequestArgs):
        try:
            import requests
            response = requests.request(
                method=args.method.upper(),
                url=args.url,
                headers=args.headers,
                json=args.data if args.data else None
            )
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text,
                "url": response.url
            }
        except Exception as e:
            return {"error": f"HTTP request failed: {str(e)}"}

    # 简单的天气查询工具（模拟）
    @api_mcp.tool(
        name="weather.query",
        description="Query weather information for a city"
    )
    def weather_query(args: WeatherArgs):
        import random
        # 模拟真实天气数据
        conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
        return {
            "city": args.city,
            "unit": args.unit,
            "temperature": round(random.uniform(-10, 40), 1),
            "humidity": random.randint(30, 90),
            "condition": random.choice(conditions),
            "wind_speed": round(random.uniform(0, 30), 1),
            "timestamp": datetime.now().isoformat()
        }


# ---------------- 启动 ----------------
if __name__ == "__main__":
    server_name = TOOL_CONFIG.get("server_info", {}).get("toolName")
    port = is_port_free(args.port)

    if port:
        print(f"Starting {TOOL_CONFIG.get('serverName', 'API Server')} on port {args.port}")

        try:
            # 创建 FastMCP（把选好的 port 传入）
            # TODO 必须指明host为0.0.0.0，否则只能localhost访问
            api_mcp = FastMCP(
                TOOL_CONFIG.get("server_info", {}).get("toolName"),
                json_response=True, host="0.0.0.0", port=args.port
            )

            # 注册工具
            register_tools(api_mcp)

            # 输出启动成功标识，用于外部检测
            print("Server startup successful, waiting for connections...")

            # 使用 streamable-http（你当前使用的方式）
            api_mcp.run("streamable-http")
        except Exception as e:
            # TODO 输出启动失败标识和错误信息，父进程会检测Server startup failed:需要明确给出
            print(f"Server startup failed: {str(e)}", file=sys.stderr)
            sys.exit(1)  # 退出进程，这将被父进程检测到
    else:
        print("No available port found, server startup failed.")
        sys.exit(1)  # 退出进程
