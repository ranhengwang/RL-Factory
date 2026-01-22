import requests
import json
import time
import os

def test_fastapi_uploadTrajectory():

    # 1. 设置请求参数
    # url = "http://127.0.0.1:8224/trajectory/uploadTrajectory"  # FastAPI 默认端口是 8224
    url = "http://192.168.1.86:8224/trajectory/uploadTrajectory"  # FastAPI 默认端口是 8224

    trajectory_data_list = [
    {
        "steps": [
            {"stepID": 1, "thought": "收到写报告的任务，首先需要列大纲", "action": "Create Outline", "observation": "大纲已生成，包含3个主要部分", "reward": 0.5},
            {"stepID": 2, "thought": "开始撰写第一部分：背景介绍", "action": "Write Introduction", "observation": "第一段落完成，字数200字", "reward": 0.8}
        ]
    },
    # 2. 格式检查 (已提供)
    {
        "steps": [
            {"stepID": 1, "thought": "检查引用格式", "action": "Check References", "observation": "发现2处格式错误", "reward": 0.3}
        ]
    },
    # 3. 代码 Debug 场景
    {
        "steps": [
            {"stepID": 1, "thought": "用户反馈登录报错，查看后端日志", "action": "Read Logs", "observation": "发现 NullPointerException 在 AuthService 24行", "reward": 0.6},
            {"stepID": 2, "thought": "分析是由于空用户对象引起的，添加空值判断", "action": "Fix Code", "observation": "测试用例通过", "reward": 1.0}
        ]
    },
    # 4. 论文检索场景
    {
        "steps": [
            {"stepID": 1, "thought": "搜索关于大语言模型长文本处理的论文", "action": "Search Google Scholar", "observation": "获得10篇相关文献，涉及RAG和FlashAttention", "reward": 0.7}
        ]
    },
    # 5. SQL 优化场景
    {
        "steps": [
            {"stepID": 1, "thought": "查询运行缓慢，检查执行计划", "action": "EXPLAIN ANALYZE", "observation": "发现全表扫描，主键索引未生效", "reward": 0.4},
            {"stepID": 2, "thought": "尝试强制指定索引", "action": "Add Index Hint", "observation": "耗时从 2s 降低至 50ms", "reward": 0.9}
        ]
    },
    # 6. 会议日程安排
    {
        "steps": [
            {"stepID": 1, "thought": "协调三个人的时间开周会", "action": "Check Calendar", "observation": "周二下午2点三方都有空", "reward": 0.5},
            {"stepID": 2, "thought": "发送会议邀约", "action": "Send Invite", "observation": "所有人已接受邀请", "reward": 1.0}
        ]
    },
    # 7. 数据清理场景
    {
        "steps": [
            {"stepID": 1, "thought": "Excel 表格中存在大量重复项", "action": "Remove Duplicates", "observation": "删除了 45 条重复记录", "reward": 0.6}
        ]
    },
    # 8. 翻译任务
    {
        "steps": [
            {"stepID": 1, "thought": "将这段英文技术文档翻译成中文", "action": "Translate text", "observation": "翻译完成，但 'Latency' 一词需校对", "reward": 0.7},
            {"stepID": 2, "thought": "根据语境将延迟统一为 '时延'", "action": "Refine Translation", "observation": "文本通顺，符合技术规范", "reward": 0.9}
        ]
    },
    # 9. 邮件回复
    {
        "steps": [
            {"stepID": 1, "thought": "客户询问产品报价单", "action": "Draft Email", "observation": "邮件草稿已写好，包含最新价目表附件", "reward": 0.8}
        ]
    },
    # 10. 文件归档
    {
        "steps": [
            {"stepID": 1, "thought": "整理项目文件夹", "action": "Move Files", "observation": "旧文档已移至 Archive 目录", "reward": 0.4}
        ]
    },
    # 11. 性能测试
    {
        "steps": [
            {"stepID": 1, "thought": "对新接口进行压力测试", "action": "Run JMeter", "observation": "QPS 达到 1000，响应时间稳定", "reward": 0.8}
        ]
    },
    # 12. 市场调研
    {
        "steps": [
            {"stepID": 1, "thought": "分析竞品功能点", "action": "Browse Competitor Web", "observation": "整理出5个核心差异点", "reward": 0.7}
        ]
    },
    # 13. UI 调整
    {
        "steps": [
            {"stepID": 1, "thought": "按钮在移动端显示不全", "action": "Update CSS", "observation": "响应式布局已修复", "reward": 0.9}
        ]
    },
    # 14. 报错排查（网络）
    {
        "steps": [
            {"stepID": 1, "thought": "无法连接远程数据库", "action": "Ping Host", "observation": "网络不通，请求超时", "reward": 0.2},
            {"stepID": 2, "thought": "检查安全组规则", "action": "Modify Security Group", "observation": "3306 端口未对内网开放，已开启", "reward": 1.0}
        ]
    },
    # 15. 知识库总结
    {
        "steps": [
            {"stepID": 1, "thought": "根据长篇文章总结核心观点", "action": "Summarize Content", "observation": "提取出3条核心结论", "reward": 0.8}
        ]
    },
    # 16. 单元测试编写
    {
        "steps": [
            {"stepID": 1, "thought": "提高计算模块的代码覆盖率", "action": "Write Unit Tests", "observation": "新增 5 个边界值测试用例", "reward": 0.7},
            {"stepID": 2, "thought": "运行测试套件", "action": "Execute Pytest", "observation": "覆盖率提升至 95%", "reward": 1.0}
        ]
    }
    ]
    # 对应 FastAPI 中的 sftType: str = Form(...)
    payload = {
        "taskType": "report_writer",
        "description": "这是一个报告写作任务,……",
        "trajectory": trajectory_data_list
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
    url = f"http://192.168.1.86:8224/trajectory/getTrajectory?taskType={taskType}&batchSize={batchSize}"
    
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

    time.sleep(1)

    test_fastapi_getTrajectory("report_writer",1)

    test_fastapi_getTrajectory("report_writer",16)

    # {
    # "status": "success",
    # "taskType": "report_writer",
    # "description": "这是一个报告写作任务,……",
    # "count": 16,
    # "data": [
    #     {
    #         "id": 16,
    #         "steps": [
    #             {
    #                 "stepID": 1,
    #                 "thought": "提高计算模块的代码覆盖率",
    #                 "action": "Write Unit Tests",
    #                 "observation": "新增 5 个边界值测试用例",
    #                 "reward": 0.7
    #             },
    #             {
    #                 "stepID": 2,
    #                 "thought": "运行测试套件",
    #                 "action": "Execute Pytest",
    #                 "observation": "覆盖率提升至 95%",
    #                 "reward": 1.0
    #             }
    #         ],
    #         "length": 2,
    #         "evaluated": 1,
    #         "created_at": "2025-12-29 07:33:06"
    #         "question": "我想 7 月 15 日從北京飛上海並預訂兩晚酒店"
    #         "golden_answers": ["已完成機票與酒店預訂"]
    #     }
    # ]
    # }


# {"status":"success",
# "taskType":"report_writer",
# "description":"这是一个报告写作任务,……",
# "count":10,
# "data":[
#     {"id":32,
#     "question":"计算模块的测试覆盖率够吗？",
#     "golden_answers":["已升至95%"],
#     "steps":[
#         {"stepID":1,
#         "thought":"提高计算模块的代码覆盖率",
#         "action":"Write Unit Tests",
#         "observation":"新增 5 个边界值测试用例",
#         "reward":0.7},
#         {"stepID":2,
#         "thought":"运行测试套件",
#         "action":"Execute Pytest",
#         "observation":"覆盖率提升至 95%",
#         "reward":1.0}
#         ],
#         "length":2,
#         "evaluated":1,
#         "created_at":"2025-12-30 02:04:52"},
#         {"id":31,"question":"这篇长文章讲了什么重点？","golden_answers":["总结出3条核心结论"],"steps":[{"stepID":1,"thought":"根据长篇文章总结核心观点","action":"Summarize Content","observation":"提取出3条核心结论","reward":0.8}],"length":1,"evaluated":1,"created_at":"2025-12-30 02:04:52"},{"id":30,"question":"数据库连不上，是挂了吗？","golden_answers":["开了3306端口，已恢复"],"steps":[{"stepID":1,"thought":"无法连接远程数据库","action":"Ping Host","observation":"网络不通，请求超时","reward":0.2},{"stepID":2,"thought":"检查安全组规则","action":"Modify Security Group","observation":"3306 端口未对内网开放，已开启","reward":1.0}],"length":2,"evaluated":1,"created_at":"2025-12-30 02:04:52"},{"id":29,"question":"手机端按钮看不全，修好了吗？","golden_answers":["布局已修复"],"steps":[{"stepID":1,"thought":"按钮在移动端显示不全","action":"Update CSS","observation":"响应式布局已修复","reward":0.9}],"length":1,"evaluated":1,"created_at":"2025-12-30 02:04:52"},{"id":28,"question":"我们和竞品的功能差在哪？","golden_answers":["整理出5个核心差异"],"steps":[{"stepID":1,"thought":"分析竞品功能点","action":"Browse Competitor Web","observation":"整理出5个核心差异点","reward":0.7}],"length":1,"evaluated":1,"created_at":"2025-12-30 02:04:52"},{"id":27,"question":"新接口的压测表现怎么样？","golden_answers":["QPS 1000，运行稳定"],"steps":[{"stepID":1,"thought":"对新接口进行压力测试","action":"Run JMeter","observation":"QPS 达到 1000，响应时间稳定","reward":0.8}],"length":1,"evaluated":1,"created_at":"2025-12-30 02:04:52"},{"id":26,"question":"项目里那些旧文件整理好了吗？","golden_answers":["已移至 Archive 目录"],"steps":[{"stepID":1,"thought":"整理项目文件夹","action":"Move Files","observation":"旧文档已移至 Archive 目录","reward":0.4}],"length":1,"evaluated":1,"created_at":"2025-12-30 02:04:52"},{"id":25,"question":"客户在要报价单，我该怎么回？","golden_answers":["草稿已写，附带报价表"],"steps":[{"stepID":1,"thought":"客户询问产品报价单","action":"Draft Email","observation":"邮件草稿已写好，包含最新价目表附件","reward":0.8}],"length":1,"evaluated":1,"created_at":"2025-12-30 02:04:52"},{"id":24,"question":"这篇技术文档能翻成中文吗？","golden_answers":["已翻好，术语已校对"],"steps":[{"stepID":1,"thought":"将这段英文技术文档翻译成中文","action":"Translate text","observation":"翻译完成，但 'Latency' 一词需校对","reward":0.7}