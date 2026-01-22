import requests
import json
import os

def test_fastapi_uploadTrajectory():

    # 1. 设置请求参数
    url = "http://127.0.0.1:8224/trajectory/uploadTrajectory"  # FastAPI 默认端口是 8224

    trajectory_data_list = [
        {
            "steps": [
                {
                    "stepID": 1, 
                    "thought": "收到写报告的任务，首先需要列大纲", 
                    "action": "Create Outline", 
                    "observation": "大纲已生成，包含3个主要部分", 
                },
                {
                    "stepID": 2, 
                    "thought": "开始撰写第一部分：背景介绍", 
                    "action": "Write Introduction", 
                    "observation": "第一段落完成，字数200字", 
                }
            ]
        },
        {
            "steps": [
                {
                    "stepID": 1, 
                    "thought": "检查引用格式", 
                    "action": "Check References", 
                    "observation": "发现2处格式错误", 
                }
            ]
        }
    ]
    # 对应 FastAPI 中的 sftType: str = Form(...)
    payload = {
        "taskType": "report_writer",
        "trajectory": trajectory_data_list,
        "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店",
        "golden_answers": ["已完成機票與酒店預訂"]
    }
    
    # 2. 发送请求
    print(f"--- 正在发送 POST 请求至 {url} ---")
    try:
        response = requests.post(url, json=payload)
        
        # 4. 解析结果
        if response.status_code == 200:
            print("\n 上传成功！服务端响应:")
            print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        else:
            print(f"\n 上传失败 (Code: {response.status_code})")
            print("错误信息:", response.text)
            
    except Exception as e:
        print(f"连接错误: {e}")

def test_fastapi_getTrajectory(taskType:str, batchSize:int):

    # 1. 设置请求参数
    url = f"http://127.0.0.1:8224/trajectory/getTrajectory?taskType={taskType}&batchSize={batchSize}"
    # url = "http://192.168.1.86:8224/trajectory/getTrajectory?taskType=report_writer&batchSize=16"
    # 2. 发送请求
    print(f"--- 正在发送 GET 请求至 {url} ---")
    try:
        response = requests.get(url)
        
        # 3. 解析结果
        if response.status_code == 200:
            print("\n 获取所有SFT信息成功！服务端响应:")
            print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        else:
            print(f"\n 获取所有SFT信息失败 (Code: {response.status_code})")
            print("错误信息:", response.text)
            
    except Exception as e:
        print(f"连接错误: {e}")

if __name__ == "__main__":
    
    test_fastapi_uploadTrajectory()

    test_fastapi_getTrajectory("report_writer",1)

    test_fastapi_getTrajectory("report_writer",2)