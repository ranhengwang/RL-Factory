import re
import torch
import string
import random
from abc import ABC
from verl import DataProto
from envs.tool_manager import TOOL_MANAGER_REGISTRY
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import tokenize_and_postprocess_data
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class Env(ABC):
    """
    RL-Factory 框架中用于强化学习环境的抽象基类。
    该类提供了环境交互、工具管理、奖励计算和数据处理的核心功能。
    它支持集中式和分布式的工具管理模式，并能够与不同的语言模型一起使用。

    Abstract base class for reinforcement learning environments in the RL-Factory framework.
    
    This class provides the core functionality for environment interaction, tool management,
    reward computation, and data processing. It supports both centralized and distributed
    tool management modes, and can work with different language models.
    
    Attributes:
        tool_manager: Manages tool execution and interaction with language models
        max_prompt_length (int): Maximum allowed length for prompts
        use_verify_tool (bool): Flag indicating whether to use verification tools
        use_process_reward (bool): Flag indicating whether to use process rewards
    """
    def __init__(self, config, centralized_actor=None):
        tool_manager_name = config.get('tool_manager')
        # Check if a tool manager is specified
        # If not, use adaptive mode
        # 如果没有指定工具管理器，使用自适应模式
        if not tool_manager_name:
            tool_manager_name = "adaptive"
        # Check if using a centralized tool manager

        # 检查是否使用集中式工具管理器
        if tool_manager_name.startswith('centralized_'):
            # 集中式模式必须提供actor
            if centralized_actor is None:
                raise ValueError(f"使用集中式工具管理器 '{tool_manager_name}' 需要提供 centralized_actor 参数")
            # 初始化集中式工具管理器
            self.tool_manager = TOOL_MANAGER_REGISTRY[tool_manager_name](
                verl_config=config, 
                centralized_actor_handle=centralized_actor
            )
        else:
            # 分布式模式
            # Distributed mode, keep original logic unchanged
            if tool_manager_name == 'adaptive':
                model_type = config.get('model_type')
                if 'qwen3' in model_type:
                    tool_manager_name = 'qwen3'
                elif 'qwen2' in model_type:
                    tool_manager_name = 'qwen2_5'
                elif'llama' in model_type:
                    tool_manager_name = 'llama3'
                else:
                    tool_manager_name = model_type
                    raise ValueError(f"'{tool_manager_name}' 需要进行适配，请添加一个对应的tool_manager")

            # 创建分布式工具管理器实例
            self.tool_manager = TOOL_MANAGER_REGISTRY[tool_manager_name](verl_config=config)

        self.max_prompt_length = config.get('max_prompt_length', 2048)
        self.use_verify_tool = False
        self.use_process_reward = config.get('use_process_reward', False)

    def verify_tool(self, data_source, solution_str, ground_truth, extra_info):
        """
        如需使用工具评估生成的响应，需修改此代码
        """
        # If you need a tool to evaluate the generated response, you need to modify the following code
        # the data would be stored in data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results']
        # print('verify tool start')
        raise NotImplementedError

    def _process_data(self, data_item, tokenizer):
        """
        将data_item处理为token并解码，这里的data_item包括提示词和响应的token IDs，以及相关的元数据
        """
        # process the data_item to the token and decode them
        # 获取提示词的token IDs
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        # 计算有效的prompt长度（排除padding）
        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        # 提取有效的prompt tokens
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        # 获取response的token IDs
        response_ids = data_item.batch['responses']

        # 计算有效的response长度（排除padding）
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        # 提取有效的response tokens
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        # 真实答案
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        # 数据来源
        data_source = data_item.non_tensor_batch['data_source']
        # 额外信息
        extra_info = data_item.non_tensor_batch.get('extra_info', None)
        
        return {
            'prompt_str': prompt_str,
            'response_str': response_str,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'extra_info': extra_info
        }
    
    def get_step_reward(self, responses, format_score=0.1):
        """
        为每个响应生成步骤奖励。
        """
        # 简单返回全1列表
        step_reward = [1] * len(responses)
    
        return step_reward

    def step(self, responses, tokenizer=None):
        """
        执行动作并获取结果
        """
        cur_actions, tool_results = self.tool_manager.execute_actions(responses=responses)
        # 初始化返回值列表
        next_obs, dones, valid_action, is_tool = [], [], [], []

        # 遍历每个动作和对应的工具结果
        for action, tool_result in zip(cur_actions, tool_results):
            if action == 'answer':
                temp_next_obs, temp_done, temp_valid_action, temp_is_tool = '', True, 1, 0
            elif action == 'error':
                # 如果提供了tokenizer，使用工具管理器格式化错误提示
                if tokenizer:
                    temp_next_obs = self.tool_manager.get_prompt(
                        input_data=tool_result, 
                        tokenizer=tokenizer,
                        mode='tool_call', 
                        add_generation_prompt=True
                    )
                else:
                    temp_next_obs = tool_result
                temp_done, temp_valid_action, temp_is_tool = False, 0, 0
            elif action == 'actions':
                if tokenizer:
                    temp_next_obs = self.tool_manager.get_prompt(
                        input_data=tool_result, 
                        tokenizer=tokenizer,
                        mode='tool_call',
                        add_generation_prompt=True
                    )
                else:
                    temp_next_obs = tool_result
                temp_done, temp_valid_action, temp_is_tool = False, 1, 1
            else:
                raise ValueError('Unexpected action: {}'.format(action))
            
            next_obs.append(temp_next_obs)
            dones.append(temp_done)
            valid_action.append(temp_valid_action)
            is_tool.append(temp_is_tool)

        return next_obs, dones, valid_action, is_tool


    def compute_score(self, reward_rollout_wg, reward_tokenizer, tokenizer, data: DataProto, if_val=False, use_process_reward=False):
        if reward_rollout_wg is not None:
            # 使用奖励模型计算分数
            scores = self._compute_score_with_reward_rollout_wg(reward_rollout_wg, reward_tokenizer, data)
        else:
            # 使用规则计算分数
            score = self._compute_score_with_rules(data, tokenizer, if_val=if_val)
            # 如果使用过程奖励且不是验证模式
            if use_process_reward and not if_val:
                scores = []
                for i in range(len(data)):
                    data_item = data[i]
                    # 获取工具使用分数
                    tool_use_score = data_item.batch['tool_use_scores']
                    # 过滤掉NaN值
                    validate_score = tool_use_score[ ~ torch.isnan(tool_use_score)].tolist()
                    # 组合工具分数和最终分数
                    scores.append(validate_score + score[i])
            else:
                scores = score
        
        return scores
    
    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # process the data_item to the token and decode them
            # 处理数据项为token并解码
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']

            # reserved for compatibility
            prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']

        scores = [[0.0]] * len(data)
        
        return scores

    def get_prompt_for_reward(self, reward_tokenizer, data: DataProto):
        """
        将模型生成的多步骤响应切分成独立的片段，并为每个片段生成用于奖励模型评估的提示词。
        reward_tokenizer: 奖励模型的分词器（可能与生成模型的分词器不同）
        data: DataProto: 包含批量数据的协议对象，里面有多个样本
        """

        # 二维列表，结构为：[[样本1的prompt列表], [样本2的prompt列表], ...]
        # 每个样本可能有多个步骤，因此需要多个prompt
        reward_prompt_strs = []
        
        for i in range(len(data)):
            data_item = data[i]
            # 是一个一维张量，标记响应中哪些位置是"步骤结束点"
            # 例如：[0, 0, 0, 1, 0, 0, 1, 0, 1]，其中1表示该位置是一个步骤的结束
            step_mask = data.batch['step_mask'][i]
            
            # 原始问题的token IDs（可能包含padding）
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            # 模型生成的响应的token IDs
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # 获取非tensor数据
            non_tensor_data = {
                'data_source': data_item.non_tensor_batch['data_source'],
                'ground_truth': data_item.non_tensor_batch['reward_model']['ground_truth'],
                'extra_info': data_item.non_tensor_batch.get('extra_info', None)
            }

            # 找到所有step的位置
            mask_indices = torch.where(step_mask == 1)[0]
            assert len(mask_indices) > 0, "no step mask"
            
            # 处理所有responses
            reward_prompt_str_list = []
            start_idx = 0
            
            for end_idx in mask_indices:
                # 截取当前response
                current_response = valid_response_ids[start_idx:end_idx]
                current_response_str = reward_tokenizer.decode(current_response, skip_special_tokens=True)
                
                # 生成prompt
                reward_prompt_str = self._get_single_prompt_str(
                    data_source=non_tensor_data['data_source'],
                    solution_str=current_response_str,
                    ground_truth=non_tensor_data['ground_truth'],
                    extra_info=non_tensor_data['extra_info'],
                    reward_tokenizer=reward_tokenizer
                )
                reward_prompt_str_list.append(reward_prompt_str)
                
                start_idx = end_idx

            assert start_idx == len(valid_response_ids) - 1, "start_idx is not the last index"

            reward_prompt_strs.append(reward_prompt_str_list)
        
        return reward_prompt_strs
    
    def _get_single_prompt_str(self, data_source, solution_str, ground_truth, extra_info, reward_tokenizer):
        # If you need use the reasoning model to generate the reward, you need to modify the following code
        # result = reward_tokenizer.apply_chat_template([
        #     {'role': 'system', 'content': 'You are a assistant. '},
        #     {'role': 'user', 'content': '你是Qwen吗？你只需要回答是或者不是即可。'}
        # ], add_generation_prompt=True, tokenize=False)
        # return result
        raise NotImplementedError

    def _compute_score_with_reward_rollout_wg(self, reward_rollout_wg, reward_tokenizer, data: DataProto):
        # 基于actor rollout的回答和真实答案构造judge model的prompts
        reward_prompt_strs = self.get_prompt_for_reward(reward_tokenizer, data)

        # 展平reward_prompt_strs为一个batch
        flat_prompts = []
        original_shapes = []  # 记录每个样本的prompt数量
        for prompts in reward_prompt_strs:
            original_shapes.append(len(prompts))
            flat_prompts.extend(prompts)

        # 将flat_prompts转换为DataProto格式
        input_ids = []
        attention_mask = []
        for prompt in flat_prompts:
            # 使用 tokenize_and_postprocess_data 处理每个 prompt
            ids, mask = tokenize_and_postprocess_data(
                prompt=prompt,
                tokenizer=reward_tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=reward_tokenizer.pad_token_id,
                left_pad=True,
                truncation='error'
            )
            input_ids.append(ids)
            attention_mask.append(mask)

        # 创建DataProto
        tensors = {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0)
        }

        # 计算position_ids
        tensors["position_ids"] = compute_position_id_with_mask(tensors["attention_mask"])
        data_proto = DataProto.from_dict(tensors=tensors)

        # padding并生成response
        size_divisor = reward_rollout_wg.world_size
        data_proto_padded, pad_size = pad_dataproto_to_divisor(data_proto, size_divisor)
        responses_data = reward_rollout_wg.generate_sequences(data_proto_padded)
        responses_data = unpad_dataproto(responses_data, pad_size=pad_size)

        # 计算每个response的分数
        flat_scores = []
        for i, temp_response_data in enumerate(responses_data):
            # 找到对应的data_item
            data_idx = 0
            prompt_count = 0
            while data_idx < len(original_shapes) and prompt_count + original_shapes[data_idx] <= i:
                prompt_count += original_shapes[data_idx]
                data_idx += 1
            
            data_item = data[data_idx]

            temp_prompt_ids = temp_response_data.batch['prompts']
            temp_prompt_length = temp_prompt_ids.shape[-1]
            temp_response_ids = temp_response_data.batch['responses']
            temp_valid_response_length = temp_response_data.batch['attention_mask'][temp_prompt_length:].sum()
            temp_valid_response_ids = temp_response_ids[:temp_valid_response_length]
            response_str = reward_tokenizer.decode(temp_valid_response_ids, skip_special_tokens=True)

            score = self._compute_single_score_with_reward_rollout_wg(
                data_item.non_tensor_batch['data_source'], 
                response_str, 
                data_item.non_tensor_batch['reward_model']['ground_truth'], 
                data_item.non_tensor_batch.get('extra_info', None)
            )
            flat_scores.append(score)
        
        assert data_idx == len(data) - 1, "data_idx is {}".format(data_idx)
        
        # 将scores重新组织为原来的形状
        scores = []
        start_idx = 0
        for shape in original_shapes:
            end_idx = start_idx + shape
            scores.append(flat_scores[start_idx:end_idx])
            start_idx = end_idx
        
        return scores
    
    def _compute_single_score_with_reward_rollout_wg(self, data_source, solution_str, ground_truth, extra_info):
        # print("solution_str: ", solution_str)
        # return 1.0
        raise NotImplementedError