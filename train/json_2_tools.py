import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


def _safe_identifier(name: str) -> str:
    """工具名/函数名转成合法 Python identifier"""
    name = (name or "").strip()
    # 把 . / - 空白 等轉 _
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not name:
        name = "unnamed_tool"
    if name[0].isdigit():
        name = "tool_" + name
    return name


def _safe_filename(name: str) -> str:
    """檔名安全化（不含副檔名）"""
    # 檔名允許多一點，但先簡化：全轉 identifier
    return _safe_identifier(name)


def _param_type_to_py(param_type: Any) -> str:
    """
    你的 mcp.json 裡 type 是中文：字符串/整数/对象/数组/布尔/数字...
    做一個簡單映射；未知就 Any
    """
    t = str(param_type or "").strip().lower()
    mapping = {
        "字符串": "str",
        "string": "str",
        "整数": "int",
        "integer": "int",
        "数字": "float",
        "number": "float",
        "布尔": "bool",
        "boolean": "bool",
        "对象": "dict",
        "object": "dict",
        "数组": "list",
        "array": "list",
    }
    return mapping.get(t, "Any")

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_server_name(mcp_json: Dict[str, Any]) -> str:
    # 你要求用 mcp.json 的 toolName 當檔名
    server_info = mcp_json.get("server_info") or {}
    tool_name = server_info.get("toolName") or "MCPServer"
    return str(tool_name)


def extract_tools(mcp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = mcp_json.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("mcp.json 的 tools 欄位不是 list")
    # 只保留 dict
    return [t for t in tools if isinstance(t, dict) and t.get("name")]

def render_tool_function(tool: Dict[str, Any]) -> str:
    raw_tool_name = str(tool.get("name") or "unnamed_tool")
    func_name = _safe_identifier(raw_tool_name)

    description = str(tool.get("description") or "No description provided.")
    returns_desc = str(tool.get("returns") or "dict: 执行结果").strip()

    params = tool.get("parameters") or []
    if not isinstance(params, list):
        params = []

    arg_defs: List[str] = []
    doc_lines: List[str] = []

    for p in params:
        if not isinstance(p, dict):
            continue
        p_name = p.get("name")
        if not p_name:
            continue
        p_name = str(p_name)

        py_type = _param_type_to_py(p.get("type"))
        required = bool(p.get("required", False))
        p_desc = str(p.get("description") or "").strip()

        if required:
            arg_defs.append(f"{p_name}: {py_type}")
        else:
            arg_defs.append(f"{p_name}: {py_type} = None")

        doc_lines.append(f"        {p_name}: {p_desc}")

    args_str = ", ".join(arg_defs)

    input_lines = []
    for p in params:
        if isinstance(p, dict) and p.get("name"):
            n = str(p["name"])
            input_lines.append(f"                '{n}': {n},")

    input_block = "\n".join(input_lines) if input_lines else ""
    doc_block = "\n".join(doc_lines) if doc_lines else "        (no parameters)"

    return f"""\
@mcp.tool()
def {func_name}({args_str}):
    \"\"\"
    {description}
    
    Args:
{doc_block}
        
    Returns:
        {returns_desc}
    \"\"\"
    try:
        # TODO: 在此处实现真实的业务逻辑
        # 当前为自动生成的 Mock 实现，仅打印参数并返回成功状态
        
        print(f"Tool '{func_name}' called.")
        
        # 模拟处理过程
        result_data = {{
            "status": "success",
            "message": f"Successfully executed tool: {func_name}",
            "input_parameters": {{
{input_block}
            }}
        }}
        
        return result_data

    except Exception as e:
       return f"⚠️ Error executing {func_name}: {{str(e)}}"
""".rstrip() + "\n"


def generate_server_file_code(server_name: str, tools: List[Dict[str, Any]]) -> str:
    """
    生成完整 server 檔案（固定 stdio）
    """
    header = """\
import json
from typing import Any, Optional, List, Dict
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务
mcp = FastMCP("LocalServer")

"""
    body_parts = [render_tool_function(t) for t in tools]
    body = "\n".join(body_parts)

    footer = """
if __name__ == "__main__":
    print("\\nStart MCP service:")
    mcp.run(transport='stdio')
"""
    return header + body + footer


def write_code_to_project_tools(project_dir: str, filename_no_ext: str, code: str) -> str:
    # tools_dir = os.path.join(project_dir, "envs", "tools")
    # os.makedirs(tools_dir, exist_ok=True)

    out_path = os.path.join(project_dir, filename_no_ext + ".py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)
    return out_path


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="讀取 mcp.json 的 tools，生成一個包含多個 @mcp.tool() 的 FastMCP server 檔案（固定 stdio）"
    )
    parser.add_argument("--mcp-json", required=True, help="mcp.json 路徑")
    parser.add_argument("--project-dir", required=True, help="project 根目錄（輸出到 {project}/envs/tools/）")
    args = parser.parse_args()

    mcp_json = load_json(args.mcp_json)
    server_tool_name = extract_server_name(mcp_json)  # 用 toolName 當檔名來源
    tools = extract_tools(mcp_json)

    if not tools:
        raise SystemExit("[ERROR] mcp.json.tools 為空，無法生成工具程式碼")

    code = generate_server_file_code(server_tool_name, tools)
    out_file = _safe_filename(server_tool_name)

    out_path = write_code_to_project_tools(args.project_dir, out_file, code)
    print(f"[OK] 已生成 MCP server 程式檔：{out_path}")
    print(f"[OK] tools 數量：{len(tools)}")


if __name__ == "__main__":
    main()

    # python -m train.json_2_tools   --mcp-json mycode/FastMCP_API_Server/mcp.json   --project-dir train/projects/my_weather_query