# mycode/download_model.py
# -------------------------------------------------
# 从模型存储 API 下载模型
# 1. 获取模型列表
# 2. 根据 modelName 查找 id
# 3. 获取下载链接并下载到本地
# -------------------------------------------------

import argparse
import os
import requests
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse


BASE_URL = "http://192.168.1.86:8080/api/v1"


def get_models_list() -> List[Dict]:
    """
    获取所有模型列表
    
    Returns:
        List[Dict]: 模型列表，如果失败返回空列表
    """
    url = f"{BASE_URL}/models"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        models = response.json()
        
        if not isinstance(models, list):
            print(f"[ERROR] API 返回格式错误，期望 list，得到 {type(models)}")
            return []
            
        print(f"[INFO] 成功获取到 {len(models)} 个模型")
        return models
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 请求模型列表失败: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] 解析模型列表失败: {e}")
        return []


def find_model_by_name(models: List[Dict], model_name: str) -> Optional[Dict]:
    """
    根据 modelName 查找模型信息
    
    Args:
        models: 模型列表
        model_name: 模型名称
        
    Returns:
        Optional[Dict]: 找到的模型信息，未找到返回 None
    """
    for model in models:
        if model.get("modelName") == model_name:
            return model
    
    print(f"[ERROR] 未找到名为 '{model_name}' 的模型")
    print(f"[INFO] 可用的模型名称列表:")
    for model in models:
        print(f"  - {model.get('modelName')} (id: {model.get('id')})")
    return None


def get_download_url(model_id: int) -> Optional[str]:
    """
    获取模型的下载链接（预签名 URL）
    
    Args:
        model_id: 模型 ID
        
    Returns:
        Optional[str]: 预签名下载链接，失败返回 None
    """
    url = f"{BASE_URL}/models/{model_id}/download"
    
    try:
        # 先不跟随重定向，检查响应类型
        response = requests.get(url, timeout=30, allow_redirects=False)
        
        # 如果是重定向（302/301），获取 Location 头中的预签名 URL
        if response.status_code in [301, 302, 303, 307, 308]:
            download_url = response.headers.get('Location')
            if download_url:
                print(f"[INFO] 获取到重定向预签名链接")
                return download_url
        
        # 如果是成功响应，尝试解析 JSON
        response.raise_for_status()
        
        # 尝试解析为 JSON
        try:
            result = response.json()
            
            # 如果返回的是字典，尝试获取 url 字段
            if isinstance(result, dict):
                download_url = result.get("url") or result.get("downloadUrl") or result.get("download_url")
                if download_url:
                    return download_url
            
            # 如果返回的是字符串，直接使用
            if isinstance(result, str):
                return result
                
        except (ValueError, requests.exceptions.JSONDecodeError):
            # 如果不是 JSON，尝试作为纯文本 URL
            text_result = response.text.strip()
            if text_result.startswith('http'):
                return text_result
        
        print(f"[WARN] 下载接口返回格式未识别: {response.text[:200]}")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 获取下载链接失败: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] 解析下载链接失败: {e}")
        return None


def download_file(url: str, save_path: str, chunk_size: int = 8192) -> bool:
    """
    下载文件到本地
    
    Args:
        url: 下载链接
        save_path: 保存路径
        chunk_size: 下载块大小
        
    Returns:
        bool: 是否成功
    """
    try:
        print(f"[INFO] 开始下载: {url}")
        print(f"[INFO] 保存到: {save_path}")
        
        # 创建目录
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        # 流式下载
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r[INFO] 下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
        
        print()  # 换行
        print(f"[OK] 下载完成: {save_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] 下载失败: {e}")
        # 删除不完整的文件
        if os.path.exists(save_path):
            os.remove(save_path)
        return False
    except Exception as e:
        print(f"\n[ERROR] 保存文件失败: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def extract_zip(zip_path: str, extract_to: Optional[str] = None, remove_zip: bool = False) -> Optional[str]:
    """
    解压 ZIP 文件到指定目录
    
    Args:
        zip_path: ZIP 文件路径
        extract_to: 解压目标目录，如果为 None 则解压到 ZIP 文件所在目录的同名文件夹
        remove_zip: 解压后是否删除 ZIP 文件
        
    Returns:
        Optional[str]: 解压后的目录路径，失败返回 None
    """
    if not os.path.exists(zip_path):
        print(f"[ERROR] ZIP 文件不存在: {zip_path}")
        return None
    
    if not zipfile.is_zipfile(zip_path):
        print(f"[ERROR] 不是有效的 ZIP 文件: {zip_path}")
        return None
    
    # 确定解压目录
    if extract_to is None:
        # 默认解压到 ZIP 文件所在目录的同名文件夹（去掉 .zip 扩展名）
        base_name = os.path.splitext(zip_path)[0]
        extract_to = base_name
    
    try:
        print(f"[INFO] 正在解压: {zip_path}")
        print(f"[INFO] 解压到: {extract_to}")
        
        # 创建解压目录
        os.makedirs(extract_to, exist_ok=True)
        
        # 解压文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取 ZIP 文件中的所有文件列表
            file_list = zip_ref.namelist()
            print(f"[INFO] ZIP 中包含 {len(file_list)} 个文件/目录")
            
            # 解压所有文件
            zip_ref.extractall(extract_to)
        
        print(f"[OK] 解压完成: {extract_to}")
        
        # 如果需要，删除 ZIP 文件
        if remove_zip:
            os.remove(zip_path)
            print(f"[INFO] 已删除 ZIP 文件: {zip_path}")
        
        return extract_to
        
    except zipfile.BadZipFile:
        print(f"[ERROR] ZIP 文件损坏: {zip_path}")
        return None
    except Exception as e:
        print(f"[ERROR] 解压失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="从模型存储 API 下载模型")
    parser.add_argument("--model-name", required=True, help="模型名称 (modelName)")
    parser.add_argument("--output-dir", default="./models", help="下载保存目录，默认 ./models")
    parser.add_argument("--base-url", default=BASE_URL, help=f"API 基础 URL，默认 {BASE_URL}")
    parser.add_argument("--extract", action="store_true", help="下载后自动解压 ZIP 文件")
    parser.add_argument("--remove-zip", action="store_true", help="解压后删除 ZIP 文件（需要 --extract）")
    args = parser.parse_args()
    
    # 1. 获取模型列表
    print(f"[INFO] 正在获取模型列表...")
    models = get_models_list()
    if not models:
        print("[ERROR] 无法获取模型列表，程序退出")
        return
    
    # 2. 根据 modelName 查找模型
    print(f"[INFO] 正在查找模型: {args.model_name}")
    model = find_model_by_name(models, args.model_name)
    if not model:
        return
    
    model_id = model.get("id")
    print(f"[INFO] 找到模型: {model.get('modelName')} (id: {model_id})")
    
    # 3. 强制通过 /download 接口获取预签名 URL（不使用模型信息中的原始 url）
    print(f"[INFO] 正在通过下载接口获取预签名链接...")
    download_url = get_download_url(model_id)
    
    if not download_url:
        print("[ERROR] 无法获取下载链接，程序退出")
        print("[INFO] 提示：必须通过 /download 接口获取预签名 URL，模型信息中的 url 字段无签名参数")
        return
    
    # 验证是否是预签名 URL（包含 X-Amz- 参数）
    if 'X-Amz-' in download_url:
        print(f"[INFO] 已获取预签名下载链接（包含签名参数）")
    else:
        print(f"[WARN] 下载链接可能不是预签名 URL，如果下载失败请检查")
    
    print(f"[INFO] 下载链接: {download_url[:100]}..." if len(download_url) > 100 else f"[INFO] 下载链接: {download_url}")
    
    # 4. 确定保存路径
    filename = model.get("filename")
    if not filename:
        # 从 URL 提取文件名（去掉查询参数）
        parsed_url = urlparse(download_url)
        filename = os.path.basename(parsed_url.path) or f"model_{model_id}.zip"
    
    model_path = model.get("path", "")
    if model_path:
        save_dir = os.path.join(args.output_dir, model_path)
    else:
        save_dir = os.path.join(args.output_dir, args.model_name)
    
    save_path = os.path.join(save_dir, filename)
    
    # 5. 下载文件
    success = download_file(download_url, save_path)
    
    if not success:
        print("[ERROR] 模型下载失败")
        return
    
    print(f"[OK] 模型下载成功!")
    print(f"  模型名称: {model.get('modelName')}")
    print(f"  版本: {model.get('version')}")
    print(f"  保存路径: {save_path}")
    print(f"  文件大小: {model.get('size', 'N/A')} MB")
    
    # 6. 如果指定了 --extract，自动解压
    if args.extract or filename.lower().endswith('.zip'):
        # 如果文件是 ZIP 格式，自动解压
        extract_dir = extract_zip(save_path, remove_zip=args.remove_zip)
        if extract_dir:
            print(f"[OK] 模型已解压到: {extract_dir}")
        else:
            print(f"[WARN] 解压失败，但 ZIP 文件已保存: {save_path}")


if __name__ == "__main__":
    main()
# # 下载并解压，解压后删除 ZIP 文件
# python download_model.py --model-name my-first-model --extract --remove-zip