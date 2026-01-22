import requests
import json
import os

def upload_agent_model(file_path, url):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    file_name = os.path.basename(file_path)
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

    print(f"Uploading {file_name} to {url}...")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")

    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_name, f, 'application/zip')
            }
            data = {
                'metadata': json.dumps(metadata)
            }
            
            response = requests.post(url, files=files, data=data)
            
            print(f"Status Code: {response.status_code}")
            try:
                print("Response JSON:", response.json())
            except:
                print("Response Text:", response.text)
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Configuration
    SERVER_URL = "http://192.168.1.86:8080/api/v1/models"
    # ZIP_FILE_PATH = "weather_query.zip" # 默认上传当前目录下的 weather_query.zip
    ZIP_FILE_PATH = "FastMCP_API_Server.zip" # 默认上传当前目录下的 weather_query.zip

    upload_agent_model(ZIP_FILE_PATH, SERVER_URL)