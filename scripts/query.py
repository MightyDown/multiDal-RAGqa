"""CLI: 查询 multiDal 知识库。"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multidal.agents.query_agent import QueryAgent
from multidal.kb.manager import KBManager
from multidal.kb.rewriter import QueryRewriter
from multidal.kb.router import IntentRouter


async def main_async(args) -> None:
    kb_mgr = KBManager()
    router = IntentRouter(kb_mgr)
    kb_ids = await router.route(args.question, args.kb or None, args.auto_route)

    queries = [args.question]
    if args.rewrite:
        rewriter = QueryRewriter()
        queries = await rewriter.rewrite(args.question)

    agent = QueryAgent()
    context = f"知识库: {kb_ids}\n改写查询: {queries}\n(检索结果待接入)"
    answer = await agent.run(args.question, context)
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query multiDal knowledge bases")
    parser.add_argument("--question", required=True, help="Your question")
    parser.add_argument("--kb", nargs="*", help="Target KB IDs")
    parser.add_argument("--auto-route", action="store_true")
    parser.add_argument("--no-rewrite", dest="rewrite", action="store_false")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
