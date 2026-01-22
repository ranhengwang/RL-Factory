# mycode/data_processing.py
# -------------------------------------------------
# 把带 trajectory 的 JSONL 转成 Parquet
# (兼容 trajectory 和 trajectory_data_list 两种字段名)
# -------------------------------------------------
import argparse
import json
import os
from datetime import datetime

import pandas as pd


DEFAULT_SYSTEM_PREFIX = (
    "You are a helpful AI assistant</think> first every time you get new information. "
    "Finally, provide the answer inside <answer> and </answer>."
)
DEFAULT_USER_PREFIX = "Question: {question}\n"


def make_prefix(question: str,
                system_prefix: str = DEFAULT_SYSTEM_PREFIX,
                user_prefix: str = DEFAULT_USER_PREFIX) -> str:
    """拼装 prompt：system + user"""
    if question[-1] != "?" and not question.endswith("？"):
        question += "?"
    return f"{system_prefix}\n\n{user_prefix.format(question=question)}"


def process_jsonl(input_path: str,
                  system_prefix: str = DEFAULT_SYSTEM_PREFIX,
                  user_prefix: str = DEFAULT_USER_PREFIX) -> list[dict]:
    """
    读取 JSONL，返回符合 RL-Factory 数据格式的 list[dict]
    """
    processed = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {idx} 解析失败：{e}")
                continue

            if "question" not in example:
                print(f"[WARN] line {idx} 缺少字段 question，跳过")
                continue
            if "golden_answers" not in example:
                print(f"[WARN] line {idx} 缺少字段 golden_answers，跳过")
                continue

            prompt_txt = make_prefix(
                example["question"].strip(), system_prefix, user_prefix
            )

            item = dict(
                data_source="nq",
                prompt=[{"role": "user", "content": prompt_txt}],
                ability="fact-reasoning",
                reward_model={
                    "style": "rule",
                    "ground_truth": {"target": example["golden_answers"]},
                },
                extra_info={"split": "train", "index": idx - 1},
            )

            # ---- 关键：把轨迹同时放到顶层和 interaction_kwargs ----
            traj_data = example.get("trajectory_data_list") or example.get("trajectory")
            if traj_data:
                # 1. 顶层字段（用于直接读取）
                item["trajectory_data_list"] = traj_data
                # 2. 复制到 interaction_kwargs（作为备用通道）
                item["extra_info"]["interaction_kwargs"] = {
                    "trajectory_data_list": traj_data
                }

            processed.append(item)
    return processed


def save_parquet(data: list[dict], output_path: str):
    """保存成 Parquet；自动建目录"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"[OK] Saved {len(df)} rows -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="JSONL ➜ Parquet (with trajectory)")
    parser.add_argument("--input", required=True, help="输入 JSONL")
    parser.add_argument("--output", required=False,
                        help="输出 Parquet；默认与输入同名 .parquet")
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        base = os.path.splitext(args.input)[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{base}_{ts}.parquet"

    data = process_jsonl(args.input)
    if not data:
        print("[ERROR] 没有有效样本，停止")
        return
    save_parquet(data, out_path)


if __name__ == "__main__":
    main()