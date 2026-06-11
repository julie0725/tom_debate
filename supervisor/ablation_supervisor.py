"""
supervisor/ablation_supervisor.py
----------------------------------
E / G 조건용 Supervisor 확장.
기존 supervisor.py 수정 없음.

CamelSupervisor   (E): 에이전트를 CAMEL 버전으로 교체
SingleAgentSupervisor (G): AgentCombined 단일 에이전트, 토론 없음
"""
import logging

from supervisor.supervisor import Supervisor

logger = logging.getLogger(__name__)


class CamelSupervisor(Supervisor):
    """
    E 조건: 기존 3-agent + debate 구조 유지,
    Agent1/2/3의 LLM 호출만 CAMEL ChatAgent로 교체.
    """

    def __init__(self, pool, config: dict):
        # 부모가 기존 Agent1/2/3 생성
        super().__init__(pool, config)
        # CAMEL 버전으로 교체
        self._swap_agents(config)

    def _swap_agents(self, config: dict) -> None:
        from agents.camel_agents import CamelAgent1, CamelAgent2, CamelAgent3

        agent_cfg = config.get("agents", {})
        new_agents = {}

        if agent_cfg.get("use_agent1", True) and 1 in self.agents:
            new_agents[1] = CamelAgent1(
                model=self.model, max_tokens=self.max_tokens,
                provider=self.provider, base_url=self.base_url,
            )
        if agent_cfg.get("use_agent2", True) and 2 in self.agents:
            new_agents[2] = CamelAgent2(
                model=self.model, max_tokens=self.max_tokens,
                provider=self.provider, base_url=self.base_url,
            )
        if agent_cfg.get("use_agent3", True) and 3 in self.agents:
            new_agents[3] = CamelAgent3(
                model=self.model, max_tokens=self.max_tokens,
                provider=self.provider, base_url=self.base_url,
            )

        self.agents = new_agents
        self.debate_manager.agents = new_agents
        logger.info(f"[CamelSupervisor] Swapped {list(new_agents.keys())} to CAMEL agents")


class SingleAgentSupervisor(Supervisor):
    """
    G 조건: AgentCombined 하나로 3역할 수행, 토론 없음.
    단일 에이전트이므로 _check_agreement() 항상 True → debate 미진입.
    """

    def __init__(self, pool, config: dict):
        # use_agent1/2/3=False이므로 부모에서 agents={} 생성됨
        super().__init__(pool, config)
        self._install_combined_agent()

    def _install_combined_agent(self) -> None:
        from agents.agent_combined import AgentCombined

        combined = AgentCombined(
            model=self.model, max_tokens=self.max_tokens,
            provider=self.provider, base_url=self.base_url,
        )
        self.agents = {1: combined}
        self.use_debate = False
        self.debate_manager.agents = {1: combined}
        logger.info("[SingleAgentSupervisor] AgentCombined installed, debate disabled")
