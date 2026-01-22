import json
import os
from typing import Any, Dict, Optional

def find_function_def(data):
    """递归查找包含 'name' 和 'parameters' 的函数定义字典"""
    if isinstance(data, dict):
        # 特征检查：通常 function definition 包含 name 和 parameters
        if "name" in data and "parameters" in data:
            return data
        # 递归查找
        for key, value in data.items():
            res = find_function_def(value)
            if res:
                return res
    return None

def generate_mcp_tool_code(schema: Dict[str, Any]) -> str:
    """
    根据提供的 JSON Schema 生成 MCP 工具代码 (Python)
    """
    
    # 1. 递归查找包含 'name' 和 'parameters' 的函数定义字典
    func_def = find_function_def(schema)
    if not func_def:
        return "# Error: 无法在 Schema 中找到有效的函数定义 (需包含 name 和 parameters)"

    # 2. 提取元数据
    tool_name = func_def.get("name", "unnamed_tool")
    description = func_def.get("description", "No description provided.")
    parameters = func_def.get("parameters", {})
    properties = parameters.get("properties", {})
    required_params = set(parameters.get("required", []))

    # 3. 类型映射 (JSON Schema type -> Python type)
    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict"
    }

    # 4. 构建参数列表和文档字符串
    args_definitions = []
    docstring_params = []
    
    for prop_name, prop_info in properties.items():
        json_type = prop_info.get("type", "string")
        py_type = type_map.get(json_type, "Any")
        prop_desc = prop_info.get("description", "")
        
        # 参数定义：如果是必填项则无默认值，否则默认为 None
        if prop_name in required_params:
            args_definitions.append(f"{prop_name}: {py_type}")
        else:
            args_definitions.append(f"{prop_name}: {py_type} = None")
            
        # 文档描述
        docstring_params.append(f"        {prop_name}: {prop_desc}")

    args_str = ", ".join(args_definitions)
    doc_str = "\n".join(docstring_params)

    # 5. 生成代码模板
    # 使用 f-string 构建最终的 Python 文件内容
    code_template = f"""import json
from typing import Any, Optional, List, Dict
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务
mcp = FastMCP("LocalServer")

@mcp.tool()
def {tool_name}({args_str}):
    \"\"\"
    {description}
    
    Args:
{doc_str}
        
    Returns:
        dict: 执行结果
    \"\"\"
    try:
        # TODO: 在此处实现真实的业务逻辑
        # 当前为自动生成的 Mock 实现，仅打印参数并返回成功状态
        
        print(f"Tool '{tool_name}' called.")
        
        # 模拟处理过程
        result_data = {{
            "status": "success",
            "message": f"Successfully executed tool: {tool_name}",
            "input_parameters": {{
"""
    
    # 将参数放入返回结果中以便调试
    for prop in properties.keys():
        code_template += f"                '{prop}': {prop},\n"

    code_template += f"""            }}
        }}
        
        return result_data

    except Exception as e:
       return f"⚠️ Error executing {tool_name}: {{str(e)}}"

if __name__ == "__main__":
    print("\\nStart MCP service:")
    mcp.run(transport='stdio')
"""

    return code_template

# --- 使用示例 ---
if __name__ == "__main__":
    # 你的输入数据 (已更新为新格式)
    TOOL_SCHEMA = {
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
    }

    # 生成代码
    generated_code = generate_mcp_tool_code(TOOL_SCHEMA)
    
    # 自动获取文件名
    func_def = find_function_def(TOOL_SCHEMA)
    if func_def:
        tool_name = func_def.get("name", "unnamed_tool")
        # 保持原有的目录结构 envs/tools/
        output_path = f"envs/tools/{tool_name}.py"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(generated_code)
        print(f"代码已保存至: {output_path}")
    else:
        print("错误: 无法从 Schema 中解析出工具名称，未保存文件。")