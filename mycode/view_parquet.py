import pandas as pd
import argparse
import sys
import os

def view_parquet(file_path):
    """Reads a Parquet file and prints its head, columns, and an example trajectory."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'")

        # 设置 pandas 显示选项以便更好地查看宽列
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1200)
        pd.set_option('display.max_colwidth', 800)

        print(f"--- Reading file: {file_path} ---")
        df = pd.read_parquet(file_path)

        print("\n--- Columns ---")
        print(df.columns.tolist())

        print("\n--- First 5 Rows ---")
        print(df.head())

        # 特别检查轨迹字段的内容
        traj_col_name = None
        if 'trajectory_data_list' in df.columns:
            traj_col_name = 'trajectory_data_list'
        elif 'trajectory' in df.columns:
            traj_col_name = 'trajectory'
        
        if traj_col_name:
            print(f"\n--- Example of '{traj_col_name}' column (first row) ---")
            # 使用 try-except 以防第一行没有轨迹数据
            try:
                # 打印第一行非空轨迹
                first_valid_traj = df[traj_col_name].dropna().iloc[0]
                print(first_valid_traj)
            except (KeyError, IndexError):
                print("Could not retrieve trajectory from the file.")
        else:
            print("\n--- Trajectory column ('trajectory_data_list' or 'trajectory') not found. ---")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View the contents of a Parquet file.")
    parser.add_argument("file_path", help="The path to the .parquet file.")
    args = parser.parse_args()
    view_parquet(args.file_path)
    # python3 view_parquet.py project/travel/data/travel_train.parquet
        # python3 view_parquet.py out.parquet