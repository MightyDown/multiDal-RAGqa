"""测试 VLM 多模态问答集成。

用法:
    # 仅测试配置和本地功能（不需要服务运行）
    python scripts/test_vlm.py --local

    # 完整测试（需要 docker-compose up -d）
    python scripts/test_vlm.py --live --host http://localhost:8000

    # 直接测试 VLMAgent 类
    python scripts/test_vlm.py --unit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_config():
    """测试 VLM 配置是否正确加载。"""
    from src.multidal.config import settings

    print("=" * 60)
    print("1. VLM Configuration")
    print("=" * 60)
    print(f"  api_base:     {settings.vlm_api_base}")
    print(f"  model:        {settings.vlm_model}")
    print(f"  max_tokens:   {settings.vlm_max_tokens}")
    print(f"  temperature:  {settings.vlm_temperature}")
    print(f"  api_key:      {'***' if settings.vlm_api_key else 'NOT SET'}")

    assert settings.vlm_model == "GLM-4.6V-Flash", f"Expected GLM-4.6V-Flash, got {settings.vlm_model}"
    assert settings.vlm_api_key, "VLM API key not configured"
    assert settings.vlm_api_base, "VLM API base not configured"
    print("  ✅ Config OK\n")


def test_image_to_base64():
    """测试本地图片转 base64。"""
    from src.multidal.agents.query_agent import VLMAgent
    from src.multidal.config import settings

    print("=" * 60)
    print("2. Image → Base64 Conversion")
    print("=" * 60)

    docs_dir = settings.project_root / "docs"
    if not docs_dir.exists():
        print("  ⚠️  docs/ directory not found, skipping")
        return

    # 找任意一张图片测试
    images = sorted(docs_dir.glob("**/*.jpg")) + sorted(docs_dir.glob("**/*.png"))
    if not images:
        print("  ⚠️  No images found in docs/, skipping")
        return

    test_img = str(images[0])
    # 模拟 retrieval result 里的 image_path 格式
    rel = test_img.replace(str(settings.project_root / "docs"), "")
    fake_path = f"/raw{rel}"

    print(f"  Test path: {fake_path}")
    b64_url = VLMAgent._image_to_base64(fake_path)
    if b64_url:
        print(f"  Base64 URL prefix: {b64_url[:60]}...")
        print(f"  ✅ Base64 conversion OK ({len(b64_url)} chars)")
    else:
        print("  ❌ Base64 conversion failed")


def test_build_messages():
    """测试 OpenAI Vision 格式的 messages 构建。"""
    from unittest.mock import MagicMock
    from src.multidal.agents.query_agent import VLMAgent

    print("\n" + "=" * 60)
    print("3. Message Building")
    print("=" * 60)

    # 创建假的 retrieval results
    text_chunk = MagicMock()
    text_chunk.modality = "text"
    text_chunk.kb_id = "kb_test"
    text_chunk.page = 3
    text_chunk.content = "Q1营收42.3亿元，同比增长18.7%。"
    text_chunk.image_path = None

    image_chunk = MagicMock()
    image_chunk.modality = "image"
    image_chunk.kb_id = "kb_test"
    image_chunk.page = 4
    image_chunk.content = "柱状图描述"
    image_chunk.image_path = "/raw/task_001/images/figure_1.jpg"

    ranked = [text_chunk, image_chunk]
    messages = VLMAgent.build_messages("Q1营收多少？", ranked)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)

    text_parts = [c for c in content if c["type"] == "text"]
    image_parts = [c for c in content if c["type"] == "image_url"]

    print(f"  Text parts:  {len(text_parts)}")
    print(f"  Image parts: {len(image_parts)}")
    print(f"  ✅ Message build OK (total content blocks: {len(content)})")


def test_has_images():
    """测试图片检测函数。"""
    from unittest.mock import MagicMock
    from src.multidal.agents.query_agent import _has_images

    print("\n" + "=" * 60)
    print("4. Image Detection (_has_images)")
    print("=" * 60)

    text_only = [MagicMock(modality="text", image_path=None) for _ in range(3)]
    assert not _has_images(text_only)
    print("  Text-only candidates → False (use text LLM)")

    mixed = text_only + [MagicMock(modality="image", image_path="/raw/task_001/images/fig.jpg")]
    assert _has_images(mixed)
    print("  Mixed candidates → True (use VLM)")

    image_no_path = [MagicMock(modality="image", image_path=None)]
    assert not _has_images(image_no_path)
    print("  Image modality but no path → False")
    print("  ✅ _has_images OK")


def test_vlm_unit():
    """单元级测试：不调真实 API。"""
    from collections import namedtuple

    print("=" * 60)
    print("5. VLMAgent Unit Test")
    print("=" * 60)

    Chunk = namedtuple("Chunk", ["modality", "kb_id", "page", "content", "image_path"])

    candidate = Chunk(
        modality="text",
        kb_id="kb_demo",
        page=2,
        content="抖音DAU超过8亿，用户日均使用时长为125分钟。",
        image_path=None,
    )

    # 仅有文本时，messages 不应包含 image_url
    from src.multidal.agents.query_agent import VLMAgent
    msgs = VLMAgent.build_messages("抖音DAU多少？", [candidate])
    img_parts = [
        c for c in msgs[0]["content"]
        if c.get("type") == "image_url"
    ]
    assert len(img_parts) == 0, "Text-only result should not contain image parts"
    print("  Text-only → no image_url blocks ✅")
    print("  ✅ Unit tests OK\n")


def test_live(host: str):
    """实时测试：对运行中的服务发请求。需要 docker-compose up -d。"""
    import requests

    print("=" * 60)
    print("Live Test")
    print("=" * 60)

    # 1. Health check
    print(f"1. Health check: {host}/api/health")
    r = requests.get(f"{host}/api/health", timeout=5)
    print(f"   Status: {r.status_code}")

    # 2. List KBs
    print(f"2. List KBs: {host}/api/kb/list")
    r = requests.get(f"{host}/api/kb/list", timeout=5)
    kbs = r.json()
    print(f"   KBs: {[kb['kb_id'] for kb in kbs] if isinstance(kbs, list) else kbs}")

    if kbs and isinstance(kbs, list) and len(kbs) > 0:
        kb_id = kbs[0]["kb_id"]

        # 3. Non-streaming query
        print(f"3. Query (non-streaming): {host}/api/query")
        r = requests.post(
            f"{host}/api/query",
            json={"question": "总结文档主要内容", "kb_ids": [kb_id], "retrieval": True, "rewrite_query": False},
            timeout=60,
        )
        data = r.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        img_sources = [s for s in sources if s.get("modality") == "image"]

        print(f"   Answer preview: {answer[:200]}...")
        print(f"   Total sources: {len(sources)}")
        print(f"   Image sources: {len(img_sources)}")
        if img_sources:
            print(f"   ✅ VLM path should have been used (images detected)")
        else:
            print(f"   ℹ️  No images → text LLM path")

        # 4. Streaming query
        print(f"4. Query (streaming): {host}/api/query/stream")
        r = requests.post(
            f"{host}/api/query/stream",
            json={"question": "总结文档主要内容", "kb_ids": [kb_id], "retrieval": True, "rewrite_query": False},
            stream=True,
            timeout=60,
        )
        chunks = []
        for line in r.iter_lines():
            if line and line.startswith("data: "):
                data_str = line[6:]
                if data_str == '{"type":"done"}':
                    break
                try:
                    event = json.loads(data_str)
                    if event.get("type") == "sources":
                        img_count = sum(1 for s in event["sources"] if s.get("modality") == "image")
                        print(f"   Sources received: {len(event['sources'])} (images: {img_count})")
                    elif event.get("type") == "delta":
                        chunks.append(event["content"])
                except Exception:
                    pass
        full = "".join(chunks)
        print(f"   Streamed answer preview: {full[:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Integration Test")
    parser.add_argument("--local", action="store_true", help="Local tests only (no server)")
    parser.add_argument("--unit", action="store_true", help="Fast unit tests")
    parser.add_argument("--live", action="store_true", help="Test against running server")
    parser.add_argument("--host", default="http://localhost:8000", help="API host")
    args = parser.parse_args()

    if args.unit:
        test_config()
        test_has_images()
        test_vlm_unit()
        print("\n🎉 All unit tests passed!")
        sys.exit(0)

    if args.local:
        test_config()
        test_image_to_base64()
        test_build_messages()
        test_has_images()
        test_vlm_unit()
        print("\n🎉 All local tests passed!")
        sys.exit(0)

    if args.live:
        test_live(args.host)
        sys.exit(0)

    # Default: local + unit
    test_config()
    test_image_to_base64()
    test_build_messages()
    test_has_images()
    test_vlm_unit()
    print("\n🎉 All tests passed!")
