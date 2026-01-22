from .base import Env as BaseEnv
from .mmbase import MMEnv
from .search import SearchEnv
from .vision import VisionEnv
from .reward_rollout_example import RewardRolloutEnv
from mycode.project.travel.envs.reward.travel import TravelEnv

# Define public interface for the module
# Specifies which classes will be imported when using "from module import *"
# from mycode.project.travel_planner.envs.reward.travel_planner import TravelPlannerEnv

# from train.projects.weather_query_agent.envs.reward.weather_query_agent import WeatherQueryAgentEnv
from train.projects.document_analysis_agent.envs.reward.document_analysis_agent import DocumentAnalysisAgentEnv
__all__ = ['BaseEnv', 'SearchEnv', 'RewardRolloutEnv', 'VisionEnv', 'MMEnv', 'TravelEnv', 'DocumentAnalysisAgentEnv']


# Environment registry mapping - connects environment names to their corresponding classes
# Facilitates dynamic environment creation by referencing names as strings
TOOL_ENV_REGISTRY = {
    'base': BaseEnv,
    'mmbase': MMEnv,
    'search': SearchEnv,
    'reward_rollout': RewardRolloutEnv,
    'vision': VisionEnv,
    'travel': TravelEnv
,
    # 'weather_query_agent': WeatherQueryAgentEnv,
    'document_analysis_agent': DocumentAnalysisAgentEnv}