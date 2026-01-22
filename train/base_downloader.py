# train/base_downloader.py
import os
import zipfile
from typing import Any, Dict, Optional
import shutil
import tempfile

import requests


class BaseDownloader:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def _get_agent_list(self) -> list:
        """获取 agent 列表"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 请求模型列表失败: {e}")
            return []
        except Exception as e:
            print(f"[ERROR] 解析模型列表失败: {e}")
            return []
    
    def _get_download_url(self, model_id: int) -> Optional[str]:
        """
        获取模型的下载链接（预签名 URL）
        
        Args:
            model_id: 模型 ID
            
        Returns:
            Optional[str]: 预签名下载链接，失败返回 None
        """
        url = f"{self.base_url}/models/{model_id}/download"
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

    def _download_file(self, url: str, save_path: str, chunk_size: int = 8192) -> bool:
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
    
    def _extract_zip(self, zip_path: str, extract_to: str) -> None:
        """解压 zip 文件，并把内容扁平化到 extract_to 目录下（去掉可能的顶层目录）"""
        os.makedirs(extract_to, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="extract_zip_") as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)

            # 过滤掉 __MACOSX 等垃圾目录
            entries = [e for e in os.listdir(tmpdir) if e not in ("__MACOSX",) and not e.startswith(".")]
            abs_entries = [os.path.join(tmpdir, e) for e in entries]

            # 若 zip 顶层只有一个目录（且没有其他文件），则把该目录内容“提一层”放进 extract_to
            if len(abs_entries) == 1 and os.path.isdir(abs_entries[0]):
                src_root = abs_entries[0]
            else:
                src_root = tmpdir

            # 把 src_root 下的内容全部移动到 extract_to（直接放在 extract_to 里）
            for name in os.listdir(src_root):
                if name in ("__MACOSX",) or name.startswith("."):
                    continue
                src = os.path.join(src_root, name)
                dst = os.path.join(extract_to, name)

                # 若目标已存在，先删除再覆盖（避免重复解压时报错）
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)

                shutil.move(src, dst)
    
    def _get_model_info(self, model_id: str) -> Dict[str, Any]:
        """获取模型信息"""
        response = requests.get(f"{self.base_url}/models/{model_id}")
        response.raise_for_status()
        return response.json()