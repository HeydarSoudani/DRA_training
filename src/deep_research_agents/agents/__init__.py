"""Deep research agents package."""

from .base_agent import BasicAgent, TagReasoningAgent
from .cpm_report import CPMReport
from .cpm_explore_agent import CPMExplore
from .research import ReSearch_Agent
from .searchr1 import SearchR1_Agent
from .stepsearch import StepSearch_Agent
from .react import ReActAgent
from .selfask import SelfAsk_Agent
from .searcho1 import SearchO1_Agent
from .webweaver_agent import WebWeaver_Agent
from .drtulu_agent import DrTulu_Agent
from .glm_agent import GLM_Agent
from .oss_agent import OSS_Agent
from .oss_bedrock_agent import OSS_BedrockAgent, BEDROCK_OSS_MODELS
from .tongyi_agent import TongyiDR_Agent
# Registry constants
REASONING_AGENTS = frozenset({
    "react",
    "selfask",
    "searcho1",
    "research",
    "searchr1",
    "stepsearch",
    "webweaver",
    "drtulu",
    "glm",
    "oss",
    "tongyi",
    "cpm_explore",
    "cpm_report",
})
CPM_REPORT_AGENTS  = frozenset()  # kept for backward-compat; now part of REASONING_AGENTS
AGENTCPM_EXPLORE_AGENTS = frozenset({"cpm_explore"})  # kept for vLLM spec resolution
ALL_AGENTS         = sorted(REASONING_AGENTS | CPM_REPORT_AGENTS | AGENTCPM_EXPLORE_AGENTS)

AGENT_MAP = {
    "react": ReActAgent,
    "selfask": SelfAsk_Agent,
    "searcho1": SearchO1_Agent,
    "research": ReSearch_Agent,
    "searchr1": SearchR1_Agent,
    "stepsearch": StepSearch_Agent,
    "webweaver": WebWeaver_Agent,
    "drtulu": DrTulu_Agent,
    "glm": GLM_Agent,
    "oss": OSS_Agent,
    "tongyi": TongyiDR_Agent,
    "cpm_explore": CPMExplore,
    "cpm_report": CPMReport,
}

__all__ = [
    "BasicAgent",
    "TagReasoningAgent",
    "CPMReport",
    "ReSearch_Agent",
    "SearchR1_Agent",
    "StepSearch_Agent",
    "ReActAgent",
    "SelfAsk_Agent",
    "SearchO1_Agent",
    "WebWeaver_Agent",
    "DrTulu_Agent",
    "GLM_Agent",
    "OSS_Agent",
    "OSS_BedrockAgent",
    "BEDROCK_OSS_MODELS",
    "TongyiDR_Agent",
    "REASONING_AGENTS",
    "CPMExplore",
    "CPM_REPORT_AGENTS",
    "AGENTCPM_EXPLORE_AGENTS",
    "ALL_AGENTS",
    "AGENT_MAP",
]
