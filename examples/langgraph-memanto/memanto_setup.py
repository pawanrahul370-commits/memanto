"""Memanto agent lifecycle helper for the LangGraph demo.

Memanto's session model requires an active session before remember/recall
calls succeed. This helper hides the create + activate + teardown
ceremony so the demo scripts can focus on the graph.
"""

from __future__ import annotations

import logging

from memanto.app.utils.errors import AgentAlreadyExistsError
from memanto.cli.client.sdk_client import SdkClient

logger = logging.getLogger(__name__)


class MemantoSetup:
    """Create an agent (idempotent), activate a session, hand back the client."""

    def __init__(self, api_key: str) -> None:
        self.client = SdkClient(api_key=api_key)

    def setup(
        self,
        agent_id: str,
        description: str | None = None,
        duration_hours: int = 6,
    ) -> SdkClient:
        """Create the agent if it doesn't exist, then activate a session."""
        try:
            self.client.create_agent(
                agent_id=agent_id,
                pattern="tool",
                description=description,
            )
            logger.info("Created Memanto agent '%s'", agent_id)
        except AgentAlreadyExistsError:
            logger.info("Memanto agent '%s' already exists, reusing", agent_id)

        self.client.activate_agent(agent_id, duration_hours=duration_hours)
        logger.info("Activated session for agent '%s'", agent_id)
        return self.client

    def teardown(self, agent_id: str) -> None:
        """Deactivate the agent session. Safe to call even if already ended."""
        try:
            self.client.deactivate_agent(agent_id)
            logger.info("Deactivated session for agent '%s'", agent_id)
        except Exception as e:
            logger.warning("Failed to deactivate agent '%s': %s", agent_id, e)
