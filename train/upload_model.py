import requests
import json
import os
from tqdm import tqdm

class TqdmUploadAdapter:
    """用于显示上传进度的适配器"""
    def __init__(self, file_path, chunk_size=8192):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.chunk_size = chunk_size
        self.progress_bar = tqdm(
            total=self.file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"上传 {os.path.basename(file_path)}"
        )
    
    def __iter__(self):
        with open(self.file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                self.progress_bar.update(len(chunk))
                yield chunk
        self.progress_bar.close()
    
    def __len__(self):
        return self.file_size

def upload_agent_model(file_path, url):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_size_gb = file_size / (1024 ** 3)
    
    # modelName 设置为压缩包名称
    model_name = os.path.splitext(file_name)[0]

    metadata = {
        "modelName": model_name,
        "version": "1.0.0",
        "description": "This is a model.",
        "baseModel": "null",
        "trainingType": "model",
        "datasetName": "null",
        "hyperparameters": "null",
        "metrics": "null",
        "status": "active",
        "creator": "admin"
    }

    print(f"准备上传文件: {file_name}")
    print(f"文件大小: {file_size_gb:.2f} GB ({file_size:,} bytes)")
    print(f"上传地址: {url}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
    print("\n开始上传...")

    try:
        # 创建流式上传适配器
        file_iterator = TqdmUploadAdapter(file_path, chunk_size=1024*1024)  # 1MB chunks
        
        files = {
            'file': (file_name, file_iterator, 'application/zip')
        }
        data = {
            'metadata': json.dumps(metadata)
        }
        
        # 设置超时时间：连接超时30秒，读取超时根据文件大小动态设置
        # 假设上传速度至少 1MB/s，为安全起见设置为文件大小(MB) * 2 秒
        read_timeout = max(3600, int(file_size / (1024 * 1024) * 2))  # 最少1小时
        
        print(f"\n超时设置: 连接=30秒, 读取={read_timeout}秒 ({read_timeout//60}分钟)")
        
        response = requests.post(
            url, 
            files=files, 
            data=data,
            timeout=(30, read_timeout),  # (connect_timeout, read_timeout)
            stream=False  # 不需要流式响应，因为我们是上传
        )
        
        print(f"\n上传完成!")
        print(f"Status Code: {response.status_code}")
        
        try:
            print("Response JSON:", json.dumps(response.json(), indent=2, ensure_ascii=False))
        except:
            print("Response Text:", response.text[:1000])  # 只打印前1000字符
            
        if response.status_code == 200 or response.status_code == 201:
            print("\n✅ 模型上传成功!")
        else:
            print(f"\n❌ 上传失败，状态码: {response.status_code}")
                
    except requests.exceptions.Timeout:
        print(f"\n❌ 上传超时! 文件太大或网络速度太慢")
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ 连接错误: {e}")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

def upload_with_curl(file_path, url):
    """
    备选方案：使用 curl 命令上传（更稳定，支持断点续传）
    需要安装 curl
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    file_name = os.path.basename(file_path)
    model_name = os.path.splitext(file_name)[0]
    
    metadata = {
        "modelName": model_name,
        "version": "1.0.0",
        "description": "This is a model.",
        "baseModel": "null",
        "trainingType": "model",
        "datasetName": "null",
        "hyperparameters": "null",
        "metrics": "null",
        "status": "active",
        "creator": "admin"
    }
    
    metadata_json = json.dumps(metadata)
    
    # 构建 curl 命令
    curl_cmd = f'''curl -X POST "{url}" \\
  -F "file=@{file_path}" \\
  -F 'metadata={metadata_json}' \\
  --progress-bar \\
  --max-time 7200 \\
  --connect-timeout 30
'''
    
    print("使用 curl 上传命令:")
    print(curl_cmd)
    print("\n执行中...")
    
    import subprocess
    result = subprocess.run(curl_cmd, shell=True)
    
    if result.returncode == 0:
        print("\n✅ 上传成功!")
    else:
        print(f"\n❌ 上传失败，返回码: {result.returncode}")

if __name__ == "__main__":
    # Configuration
    SERVER_URL = "http://192.168.1.86:8080/api/v1/models"
    # ZIP_FILE_PATH = "weather_query.zip" # 默认上传当前目录下的 weather_query.zip
    ZIP_FILE_PATH = "FastMCP_API_Server.zip" # 默认上传当前目录下的 weather_query.zip
    
    # 检查文件是否存在
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"❌ 文件不存在: {ZIP_FILE_PATH}")
        exit(1)
    
    # 方案1: 使用 requests (带进度条)
    print("=" * 60)
    print("使用 Python requests 上传 (方案1)")
    print("=" * 60)
    upload_agent_model(ZIP_FILE_PATH, SERVER_URL)
    
    # 如果方案1失败，可以尝试方案2
    # print("\n" + "=" * 60)
    # print("使用 curl 上传 (方案2 - 备选)")
    # print("=" * 60)
    # upload_with_curl(ZIP_FILE_PATH, SERVER_URL)