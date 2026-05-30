from agents import Agent
import configs
import os
import yaml


class KeywordsAgent(Agent):
    """关键词代理。"""
    def __init__(self):
        agent = Agent(
            name = "keywordsAgent",
            model = "",
            instructions="""
            You are a keyword extractor. Your task is to extract keywords from the given text.
            You should return a list of keywords.
            """
        )

    async def run(self, text: str) -> list[str]:
        """运行代理。"""
        result = await self.agent.run(text)
        return result.split("\n")