import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


DEFAULT_SYSTEM_PREFIX = (
    "You are a helpful AI assistant</think</think> first every time you get new information. "
    "Finally, provide the answer inside <answer> and </answer>."
)
DEFAULT_USER_PREFIX = "Question: {question}\n"


def make_prefix(
    question: str,
    system_prefix: str = DEFAULT_SYSTEM_PREFIX,
    user_prefix: str = DEFAULT_USER_PREFIX,
) -> str:
    """拼裝 prompt：system + user"""
    if question and question[-1] != "?" and not question.endswith("？"):
        question += "?"
    return f"{system_prefix}\n\n{user_prefix.format(question=question)}"


def fetch_from_api(url: str, task_type: str, batch_size: int, timeout: int = 60) -> List[Dict[str, Any]]:
    """
    從 API 拉取資料。期望回傳格式：
    {
      "status": "success",
      "taskType": "...",
      "count": ...,
      "data": [ { "id":..., "steps":[...], "question":..., "golden_answers":[...] }, ... ]
    }
    """
    url = (
        f"{url}/trajectory/getTrajectory"
        f"?taskType={task_type}&batchSize={batch_size}"
    )
    # url = "http://127.0.0.1:8000/trajectory/getTrajectory"

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        print(f"[ERROR] API 請求/解析失敗: {e}")
        return []

    if payload.get("status") != "success":
        print(f"[ERROR] API status != success: {payload}")
        return []

    # 注意：API 用 data 欄位承載列表
    data_list = payload.get("data")
    if not isinstance(data_list, list):
        print("[ERROR] API payload.data 不是 list")
        return []

    return data_list


def process_api_items(
    api_items: List[Dict[str, Any]],
    system_prefix: str = DEFAULT_SYSTEM_PREFIX,
    user_prefix: str = DEFAULT_USER_PREFIX,
) -> List[Dict[str, Any]]:
    """
    把 API 的 item 對齊成原先 RL-Factory Parquet 欄位格式：
    - prompt
    - reward_model.ground_truth.target
    - trajectory_data_list (對齊原本 trajectory/trajectory_data_list)
    - extra_info.interaction_kwargs.trajectory_data_list
    """
    processed: List[Dict[str, Any]] = []

    for idx, example in enumerate(api_items, 1):
        question = example.get("question")
        golden_answers = example.get("golden_answers")

        if not question:
            print(f"[WARN] item {idx} 缺少 question，跳過")
            continue
        if golden_answers is None:
            print(f"[WARN] item {idx} 缺少 golden_answers，跳過")
            continue

        prompt_txt = make_prefix(str(question).strip(), system_prefix, user_prefix)

        item: Dict[str, Any] = dict(
            data_source="nq",
            prompt=[{"role": "user", "content": prompt_txt}],
            ability="fact-reasoning",
            reward_model={
                "style": "rule",
                "ground_truth": {"target": golden_answers},
            },
            extra_info={"split": "train", "index": idx - 1},
        )

        # API 內的 steps 對齊成 trajectory_data_list
        steps = example.get("steps")
        if steps:
            item["trajectory_data_list"] = steps
            item["extra_info"]["interaction_kwargs"] = {"trajectory_data_list": steps}

        processed.append(item)

    return processed


def save_parquet(data: List[Dict[str, Any]], output_path: str) -> None:
    """保存成 Parquet；自動建目錄"""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    print(f"[OK] Saved {len(df)} rows -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="API ➜ Parquet (align trajectory_data_list)")
    parser.add_argument("--taskType", required=True, default="report_writer", help="taskType")
    parser.add_argument("--batchSize", type=int, default=16, help="batchSize")
    parser.add_argument(
        "--output",
        required=False,
        help="輸出 Parquet；預設: {taskType}_{timestamp}.parquet",
    )
    parser.add_argument("--timeout", type=int, default=60, help="API timeout seconds")
    args = parser.parse_args()

    api_items = fetch_from_api(args.taskType, args.batchSize, timeout=args.timeout)
    if not api_items:
        print("[ERROR] 沒有拿到任何資料，停止")
        return

    data = process_api_items(api_items)
    if not data:
        print("[ERROR] 沒有有效樣本，停止")
        return

    out_path = args.output
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{args.taskType}_{ts}.parquet"

    save_parquet(data, out_path)


if __name__ == "__main__":
    main()
    # python process_trajectory.py --taskType report_writer --batchSize 16 --output out.parquet