import re
import json
import torch
import random
import logging
import os
from envs.base import Env

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class WeatherQueryAgentEnv(Env):
    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)

        # ---- 规则奖励相关 ----
        # 根据 mcp.json 配置的所有工具及其必需参数
        self.tool_schemas = {
            "api_status": [],  # 无必需参数
            "math_calculate": ["expression"],  # precision 是可选参数
            "file_operation": ["operation", "path"],  # content 是可选参数
            "system_info": ["info_type"],
            "http_request": ["method", "url"],  # headers, data 是可选参数
            "weather_query": ["city"],  # unit 是可选参数
        }
        self.target_tools = list(self.tool_schemas.keys())

        # ---- 轨迹 shaping 相关 ----
        self.current_trajectories = None  # list[trajectory] or None
        self._traj_step_ptr = None  # list[int] or None

        # 建议默认稍微保守一点，避免轨迹惩罚太强导致训练抖动
        self.traj_alpha = float(config.get("traj_alpha", 0.3))

        # action -> tool alias（根据所有工具扩充）
        self.action_tool_alias = {
            # api_status
            "api": "api_status",
            "api_status": "api_status",
            "status": "api_status",
            "检查状态": "api_status",
            "服务器状态": "api_status",
            # math_calculate
            "math": "math_calculate",
            "math_calculate": "math_calculate",
            "calculate": "math_calculate",
            "计算": "math_calculate",
            "数学": "math_calculate",
            "算": "math_calculate",
            # file_operation
            "file": "file_operation",
            "file_operation": "file_operation",
            "operation": "file_operation",
            "文件": "file_operation",
            "读取": "file_operation",
            "写入": "file_operation",
            "写文件": "file_operation",
            "读文件": "file_operation",
            # system_info
            "system": "system_info",
            "system_info": "system_info",
            "info": "system_info",
            "系统": "system_info",
            "系统信息": "system_info",
            # http_request
            "http": "http_request",
            "http_request": "http_request",
            "request": "http_request",
            "请求": "http_request",
            "http请求": "http_request",
            "api请求": "http_request",
            # weather_query
            "weather": "weather_query",
            "weather_query": "weather_query",
            "query weather": "weather_query",
            "check weather": "weather_query",
            "get weather": "weather_query",
            "forecast": "weather_query",
            "天气": "weather_query",
            "气温": "weather_query",
            "温度": "weather_query",
            "查询天气": "weather_query",
            "天气查询": "weather_query",
        }

    # =============================
    # 轨迹注入（由 rollout 侧调用）
    # =============================
    def set_current_trajectories(self, traj_batch):
        """
        traj_batch: 一批样本对应的 trajectory
          - None
          - numpy object array（每个元素是 list/dict）
          - python list（每个元素是 list/dict）
          
        数据格式：trajectory_data_list 是列表，每个元素是字典，包含：
        - action: 动作名称
        - observation: 观察结果
        - reward: 奖励值
        - stepID: 步骤ID
        - thought: 思考过程
        """
        logger.warning(f"set_current_trajectories called with type: {type(traj_batch)}")

        if traj_batch is None:
            logger.warning("Received None trajectory batch")
            self.current_trajectories = None
            self._traj_step_ptr = None
            return

        # 尝试转换为列表
        try:
            # 如果是 numpy object array，需要特殊处理
            if hasattr(traj_batch, 'tolist'):
                traj_list = traj_batch.tolist()
            else:
                traj_list = list(traj_batch)
            logger.warning(f"Converted trajectory batch to list with {len(traj_list)} items")
        except Exception as e:
            logger.warning(f"Could not convert trajectory batch to list: {e}")
            traj_list = traj_batch

        self.current_trajectories = traj_list
        self._traj_step_ptr = [0 for _ in range(len(traj_list))]
        logger.warning(f"Initialized trajectory step pointers: {self._traj_step_ptr}")

        # 打印第一条轨迹的摘要信息
        if traj_list and len(traj_list) > 0:
            first_traj = traj_list[0]
            logger.warning("First trajectory type: %s", type(first_traj).__name__)
            if hasattr(first_traj, "__len__"):
                logger.warning("First trajectory length: %d", len(first_traj))
            # 打印第一条轨迹的第一个步骤（如果存在）
            if isinstance(first_traj, list) and len(first_traj) > 0:
                first_step = first_traj[0]
                if isinstance(first_step, dict):
                    logger.warning("First step keys: %s", list(first_step.keys()))

    # =============================
    # 内部工具：轨迹抽取/对齐评分
    # =============================
    def _extract_steps_from_trajectory(self, traj):
        """
        从trajectory中提取steps列表
        支持多种轨迹格式：
        1) traj = [{"action": "...", "observation": "...", "stepID": ..., ...}, ...]  
           （新格式：直接是步骤列表，适配当前数据格式）
        2) traj = [{"steps": [...]}, {"steps": [...]}]   （多段格式）
        3) traj = {"steps": [...]}                         （单段格式）
        返回：扁平化 steps list（可能为空）
        """
        if traj is None:
            return []

        # case 1: 新格式 - 直接是步骤列表（适配你的数据格式）
        if isinstance(traj, list):
            # 检查第一个元素是否是包含 action/stepID 的字典（新格式）
            if len(traj) > 0 and isinstance(traj[0], dict):
                if "action" in traj[0] or "stepID" in traj[0]:
                    # 这是新格式：直接返回
                    return traj
                # 否则可能是旧格式的 {"steps": [...]}
                elif "steps" in traj[0]:
                    # case 2: 多段格式
                    flat = []
                    for seg in traj:
                        if isinstance(seg, dict) and isinstance(seg.get("steps"), list):
                            flat.extend(seg["steps"])
                    return flat

        # case 3: 单段格式
        if isinstance(traj, dict):
            if "steps" in traj:
                steps = traj.get("steps", [])
                return steps if isinstance(steps, list) else []
            # 如果字典本身就是一个步骤（包含 action 等字段）
            elif "action" in traj or "stepID" in traj:
                return [traj]

        return []

    def _safe_lower(self, x):
        return x.lower() if isinstance(x, str) else ""

    def _normalize_tool_name(self, name: str):
        """
        把各种形态的 tool name 归一化成基础工具名，例如：
        - "weather_query-weather_query" -> "weather_query"
        - "Weather Query" -> "weather_query"
        - "api_status" -> "api_status"
        - None/<empty>/<error> -> None
        """
        if not isinstance(name, str) or not name:
            return None
        if name in ("<empty>", "<error>"):
            return None

        n = name.strip().lower()

        # 如果是 "weather_query-weather_query" 这种，取第一段
        if "-" in n:
            n = n.split("-", 1)[0].strip()

        # 如果是 "xxx.yyy" 这种，保留点号（api_status, weather_query等）
        if "." in n:
            parts = n.split(".")
            n = ".".join([p.strip() for p in parts])

        # 空格转下划线
        n = re.sub(r"\s+", "_", n)

        # 只保留常见字符（包括点号）
        n = re.sub(r"[^0-9a-z_.]+", "", n)

        return n or None

    def _tokenize_simple(self, text: str):
        """
        很轻量 token 化：英文按词，中文按字，做 Jaccard 相似度用。
        """
        text = self._safe_lower(text)
        text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
        parts = text.split()
        zh_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        return set(parts + zh_chars)

    def _infer_expected_tool_from_action(self, action_text: str):
        """
        从轨迹里执行的动作，推断期望使用的工具。
        更鲁棒的 action->tool 推断：
        1) 常见 action 直接匹配（覆盖所有工具）
        2) action 本身像 tool name 时直接识别
        3) 最后走关键词 alias
        """
        if not action_text:
            return None

        a = self._safe_lower(action_text).strip()
        a_norm = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", a).strip()

        # 直接匹配常见 action（覆盖所有工具）
        direct_map = {
            # api_status
            "api status": "api_status",
            "check status": "api_status",
            "status": "api_status",
            # math_calculate
            "calculate": "math_calculate",
            "math calculate": "math_calculate",
            "compute": "math_calculate",
            # file_operation
            "file operation": "file_operation",
            "read file": "file_operation",
            "write file": "file_operation",
            "file": "file_operation",
            # system_info
            "system info": "system_info",
            "get system": "system_info",
            "system": "system_info",
            # http_request
            "http request": "http_request",
            "make request": "http_request",
            "request": "http_request",
            # weather_query
            "query weather": "weather_query",
            "check weather": "weather_query",
            "get weather": "weather_query",
            "weather query": "weather_query",
            "weather": "weather_query",
            "forecast": "weather_query",
        }
        if a_norm in direct_map:
            return direct_map[a_norm]

        # action 自己就是 tool 名（或近似）
        a_toolish = self._normalize_tool_name(a)
        if a_toolish in self.target_tools:
            return a_toolish

        # alias（按长度倒序，优先匹配更长的关键词）
        sorted_aliases = sorted(self.action_tool_alias.items(), key=lambda x: len(x[0]), reverse=True)
        for k, v in sorted_aliases:
            if k in a:
                return v

        return None

    def _traj_alignment_score(self, sample_idx: int, model_action: str, model_tool_list):
        """
        根据当前 sample 的 trajectory step，对模型输出 action/tool 做对齐打分。
        返回范围建议 [-1, 1]。
        """
        # debug（抽样，避免刷屏）
        if random.random() < 1 / 128:
            logger.warning(
                f"Trajectory alignment - Sample {sample_idx}: "
                f"model_action='{model_action}', model_tool_list={model_tool_list}"
            )

        if self.current_trajectories is None or self._traj_step_ptr is None:
            if random.random() < 1 / 128:
                logger.warning("No trajectories loaded or step pointer not initialized")
            return 0.0

        if sample_idx >= len(self.current_trajectories):
            if random.random() < 1 / 128:
                logger.warning(
                    f"Sample index {sample_idx} out of bounds for trajectories (len={len(self.current_trajectories)})"
                )
            return 0.0

        # 从轨迹步骤中获取专家的 action 描述
        traj = self.current_trajectories[sample_idx]
        steps = self._extract_steps_from_trajectory(traj)
        if not steps:
            if random.random() < 1 / 128:
                logger.warning(f"No steps found in trajectory for sample {sample_idx}")
            return 0.0

        ptr = self._traj_step_ptr[sample_idx]
        if ptr >= len(steps):
            return 0.0

        # 适配新格式：直接访问 action 字段
        cur_step = steps[ptr] if isinstance(steps[ptr], dict) else {}
        expected_action_text = cur_step.get("action", "") or ""
        # 通过轨迹文本推断出期望使用的工具名
        expected_tool = self._infer_expected_tool_from_action(expected_action_text)
        expected_tool = self._normalize_tool_name(expected_tool) if expected_tool else None

        # 模型实际工具：取第一个有效工具名
        model_tool = None
        if isinstance(model_tool_list, list):
            for t in model_tool_list:
                if not isinstance(t, dict):
                    continue
                raw_name = t.get("name")
                norm_name = self._normalize_tool_name(raw_name)
                if norm_name:
                    model_tool = norm_name
                    break

        # 1) 工具名对齐（惩罚更柔和）
        tool_match = 0.0
        if expected_tool is not None and model_tool is not None:
            if expected_tool == model_tool:
                tool_match = 1.0
            else:
                tool_match = -0.5  # 从 -1.0 调柔
            if random.random() < 1 / 128:
                logger.warning(
                    f"Tool matching - Expected: {expected_tool}, Got: {model_tool}, "
                    f"Match: {expected_tool == model_tool}, Score: {tool_match}"
                )
        elif expected_tool is not None and model_tool is None:
            tool_match = -0.3
        else:
            tool_match = 0.0

        # 2) 文本相似度（action vs tool_name/模型动作）
        model_action_text = model_tool if model_tool else (model_action or "")
        a = self._tokenize_simple(expected_action_text)
        b = self._tokenize_simple(model_action_text)

        if len(a) == 0 and len(b) == 0:
            sim = 0.0
        else:
            sim = len(a & b) / max(1, len(a | b))

        text_score = 2.0 * sim - 1.0  # [-1, 1]

        # 工具对齐更重要，文本只做辅助（更稳）
        score = 0.8 * tool_match + 0.2 * text_score

        # 推进指针（每次 step reward 都视为推进一个轨迹 step）
        self._traj_step_ptr[sample_idx] = ptr + 1

        # clip
        score = max(-1.0, min(1.0, score))

        if random.random() < 1 / 128:
            logger.warning(
                f"Trajectory alignment score - Sample {sample_idx}, Step {ptr}: "
                f"Expected action: '{expected_action_text}', "
                f"Model action: '{model_action_text}', "
                f"Tool match: {tool_match:.2f}, Text similarity: {sim:.2f}, Final score: {score:.2f}"
            )

        return float(score)

    # =============================
    # Step Reward（规则 + 轨迹对齐融合）
    # =============================
    def get_step_reward(self, responses, format_score=0.1, tool_score=0.5):
        """
        每一步的中间奖励 (Process/Step Reward)。
        返回 list，长度 = batch_size。
        """
        step_reward = []

        # debug（抽样打印，避免刷屏）
        if random.random() < 1 / 32:
            logger.warning(f"get_step_reward called with {len(responses)} responses")
            for i, resp in enumerate(responses[:2]):
                logger.warning(f"Response {i} (first 120 chars): {str(resp)[:120]}...")
        # responses 是一个 batch，包含多个样本在当前这一步的响应
        # 每个样本独立进行多轮交互（ReAct流程）
        for i, response in enumerate(responses):
            temp_action, temp_tool_list = self.tool_manager.parse_response(response_content=response)

            # 直接回答：过程奖励 NaN
            if temp_action == "answer":
                if random.random() < 1 / 64:
                    logger.warning(f"Sample {i}: Direct answer detected, skipping step reward")
                step_reward.append(torch.nan)
                continue

            # 空调用：规则惩罚 + 轨迹对齐（但一般轨迹也给不出强信号）
            if (not temp_tool_list) or (len(temp_tool_list) > 0 and temp_tool_list[0].get("name") == "<empty>"):
                rule_score = -1.0 * format_score
                traj_score = self._traj_alignment_score(i, temp_action, temp_tool_list)
                final = (1.0 - self.traj_alpha) * rule_score + self.traj_alpha * traj_score

                if random.random() < 1 / 64:
                    logger.warning(
                        f"Sample {i}: Empty tool call - Rule score: {rule_score:.3f}, "
                        f"Traj score: {traj_score:.3f}, Final: {final:.3f}"
                    )

                step_reward.append(final)
                continue

            # ---- 规则奖励：格式 + 目标工具 + 参数完整 ----
            valid_tool_count = 0
            error_count = 0
            param_penalty = 0.0

            for tool in temp_tool_list:
                tool_name_raw = tool.get("name")
                tool_name = self._normalize_tool_name(tool_name_raw)  # 归一化
                tool_args = tool.get("arguments", tool.get("args", {}))  # 兼容 arguments/args

                # 格式错误
                if tool_name_raw == "<error>" or tool_name is None:
                    if tool_name_raw == "<error>":
                        error_count += 1
                    continue

                # 非目标工具忽略
                if tool_name not in self.target_tools:
                    continue

                required_args = self.tool_schemas[tool_name]

                # tool_args 可能是字符串（某些 tool_manager 会把 args 存成 JSON 字符串）
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        tool_args = {}

                # 对于无必需参数的工具（如 api_status），只要工具名正确就算有效
                if len(required_args) == 0:
                    if isinstance(tool_args, dict):
                        valid_tool_count += 1
                    else:
                        param_penalty -= 0.2 * tool_score
                elif isinstance(tool_args, dict) and all(k in tool_args for k in required_args):
                    valid_tool_count += 1
                else:
                    # 参数不完整：轻惩罚
                    param_penalty -= 0.2 * tool_score

            # 格式分
            if len(temp_tool_list) > 0:
                format_rew = ((len(temp_tool_list) - error_count) / len(temp_tool_list)) * format_score
            else:
                format_rew = 0.0

            task_rew = valid_tool_count * tool_score
            penalty = error_count * 0.1

            rule_score = format_rew + task_rew + param_penalty - penalty

            # ---- 轨迹对齐 shaping ----
            traj_score = self._traj_alignment_score(i, temp_action, temp_tool_list)

            final_step_score = (1.0 - self.traj_alpha) * rule_score + self.traj_alpha * traj_score
            step_reward.append(final_step_score)

            if random.random() < 1 / 128:
                logger.warning(
                    f"[traj_debug] i={i} rule={float(rule_score):.3f} traj={float(traj_score):.3f} final={float(final_step_score):.3f} "
                    f"alpha={self.traj_alpha}"
                )

        return step_reward

    # =============================
    # Final Reward（保持原逻辑）
    # =============================
    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        """
        最终奖励：只要能输出 <answer> 就算跑通（+格式分）
        """

        def check_alternate_tags(text, tag_pattern):
            matches = re.findall(tag_pattern, text)
            if not matches:
                return False
            match = re.match(r"<\/?(\w+)>", matches[0])
            if not match:
                return False
            tagname = match.group(1)
            open_tag = f"<{tagname}>"
            close_tag = f"</{tagname}>"
            stack = []
            for tag in matches:
                if tag == open_tag:
                    if stack:
                        return False
                    stack.append(tag)
                elif tag == close_tag:
                    if not stack:
                        return False
                    stack.pop()
            return len(stack) == 0

        def extract_solution(solution_str):
            think_pattern = r'`<think>`.*?`</think>`'
            solution_str = re.sub(think_pattern, "", solution_str, flags=re.DOTALL)
            answer_pattern = r"<answer>(.*?)</answer>"
            matches = re.findall(answer_pattern, solution_str, re.DOTALL)
            return matches[-1].strip() if matches else None

        format_score = 0.0 if if_val else 0.1
        scores = []

        for i in range(len(data)):
            data_item = data[i]
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            response_str = processed_data["response_str"]

            answer = extract_solution(response_str)

            answer_format_ok = check_alternate_tags(response_str, r"</?answer>")
            tool_format_ok = check_alternate_tags(response_str, r"</?tool_call>")

            current_format_score = 0.0
            current_format_score += format_score if answer_format_ok else -format_score
            current_format_score += format_score if tool_format_ok else -0.5 * format_score

            if answer is not None:
                scores.append([1.0 + current_format_score])
            else:
                scores.append([current_format_score])

        return scores
#test