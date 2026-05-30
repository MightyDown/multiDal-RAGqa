"""multiDal 集成测试入口。

测试场景：
  1. 健康检查 — 确认服务启动
  2. 知识库 CRUD — 创建 / 列表 / 删除
  3. PDF 上传 — 提交文件，获取 task_id
  4. 任务状态查询 — 轮询解析进度
  5. RAG 问答 — 提交问题，获取答案

用法:
  python main_demo.py                # 使用 TestClient 进程内测试（默认）
  python main_demo.py --live         # 对 http://localhost:8000 做实时测试
  python main_demo.py --host http://xxx:8000 --live
"""

from __future__ import annotations

import argparse
import io
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# ── 测试工具函数 ──────────────────────────────────────────────────────────

def _binfo(msg: str) -> str:
    return f"\033[1;34m{msg}\033[0m"


def _bok(msg: str) -> str:
    return f"\033[92m  PASS  {msg}\033[0m"


def _bfail(msg: str) -> str:
    return f"\033[91m  FAIL  {msg}\033[0m"


def _bwarn(msg: str) -> str:
    return f"\033[93m  WARN  {msg}\033[0m"


def _h1(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. TestClient 模式（进程内） ──────────────────────────────────────────

def _run_testclient() -> int:
    from fastapi.testclient import TestClient

    from src.multidal.api.app import app
    from src.multidal.db.models import init_db

    init_db()

    client = TestClient(app)
    failures = 0

    # 1.1 健康检查：访问 openapi.json
    _h1("1. 健康检查")
    try:
        r = client.get("/openapi.json")
        if r.status_code == 200:
            print(_bok(f"服务正常 — {r.json()['info']['title']} v{r.json()['info']['version']}"))
        else:
            print(_bfail(f"openapi.json 返回 {r.status_code}"))
            failures += 1
    except Exception as e:
        print(_bfail(f"服务不可达: {e}"))
        return 1

    # 1.2 KB 创建
    _h1("2. 知识库 CRUD")
    kb_name = f"demo_kb_{uuid.uuid4().hex[:6]}"

    try:
        r = client.post("/kb/create", json={"name": kb_name, "description": "集成测试知识库"})
        if r.status_code == 200:
            data = r.json()
            kb_id = data["kb_id"]
            print(_bok(f"创建 KB: {kb_id} ({kb_name})"))
        else:
            print(_bfail(f"创建 KB 失败: {r.status_code} {r.text}"))
            failures += 1
            kb_id = None
    except Exception as e:
        print(_bfail(f"创建 KB 异常: {e}"))
        failures += 1
        kb_id = None

    # 1.3 KB 列表
    try:
        r = client.get("/kb/list")
        if r.status_code == 200:
            data = r.json()
            print(_bok(f"KB 列表: {data['total']} 个知识库"))
            for kb in data.get("kbs", []):
                print(f"      - {kb['kb_id']}: {kb['name']} (文档数: {kb['doc_count']})")
        else:
            print(_bfail(f"KB 列表失败: {r.status_code}"))
            failures += 1
    except Exception as e:
        print(_bfail(f"KB 列表异常: {e}"))
        failures += 1

    # 1.4 PDF 上传
    _h1("3. PDF 上传")

    # 构造一个最小的合法 PDF 文件
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF"
    )

    use_kb = kb_id or "default"
    try:
        r = client.post(
            "/ingest",
            files={"file": ("demo.pdf", io.BytesIO(minimal_pdf), "application/pdf")},
            data={"kb_id": use_kb},
        )
        if r.status_code == 200:
            data = r.json()
            task_id = data["task_id"]
            print(_bok(f"上传成功: task_id={task_id}, kb={use_kb}"))
        else:
            print(_bwarn(f"上传返回 {r.status_code}: {r.text}（Kafka 可能未启动）"))
            task_id = None
    except Exception as e:
        print(_bwarn(f"上传异常: {e}（Kafka 可能未启动）"))
        task_id = None

    # 1.5 任务状态查询
    _h1("4. 任务状态查询")
    if task_id:
        try:
            r = client.get(f"/ingest/{task_id}")
            if r.status_code == 200:
                data = r.json()
                print(_bok(f"任务 {task_id}: status={data['status']}, stage={data.get('stage', 'N/A')}"))
            elif r.status_code == 404:
                print(_bwarn(f"任务 {task_id} 不存在（DB 未初始化？）"))
                failures += 1
            else:
                print(_bfail(f"状态查询失败: {r.status_code}"))
                failures += 1
        except Exception as e:
            print(_bfail(f"状态查询异常: {e}"))
            failures += 1
    else:
        print(_bwarn("跳过（无 task_id）"))

    # 1.6 RAG 问答
    _h1("5. RAG 问答")
    try:
        r = client.post(
            "/query",
            json={
                "question": "multiDal 是什么系统？",
                "kb_ids": [use_kb] if kb_id else [],
                "auto_route": False,
                "rewrite_query": False,
            },
        )
        if r.status_code == 200:
            data = r.json()
            answer_preview = data["answer"][:120]
            sources_count = len(data.get("sources", []))
            print(_bok(f"问答成功: answer='{answer_preview}...', sources={sources_count}"))
        else:
            print(_bwarn(f"问答返回 {r.status_code}: {r.text}（LLM 可能未配置）"))
    except Exception as e:
        print(_bwarn(f"问答异常: {e}（LLM / Milvus 可能未启动）"))

    # 1.7 清理
    _h1("6. 清理")
    if kb_id:
        try:
            r = client.delete(f"/kb/{kb_id}")
            if r.status_code == 200:
                print(_bok(f"已删除 KB: {kb_id}"))
            else:
                print(_bwarn(f"删除 KB 返回 {r.status_code}: {r.text}"))
        except Exception as e:
            print(_bfail(f"删除 KB 异常: {e}"))
            failures += 1

    return failures


# ── 2. Live 模式（对运行中的服务） ─────────────────────────────────────────

def _run_live(host: str) -> int:
    import requests

    failures = 0
    base = host.rstrip("/")

    # 2.1 健康检查
    _h1("1. 健康检查")
    try:
        r = requests.get(f"{base}/openapi.json", timeout=5)
        if r.status_code == 200:
            print(_bok(f"服务正常 — {r.json()['info']['title']}"))
        else:
            print(_bfail(f"服务返回 {r.status_code}"))
            return 1
    except Exception as e:
        print(_bfail(f"服务不可达: {e}"))
        return 1

    # 2.2 KB CRUD
    _h1("2. 知识库 CRUD")
    kb_name = f"demo_kb_{uuid.uuid4().hex[:6]}"
    kb_id = None

    try:
        r = requests.post(f"{base}/kb/create", json={"name": kb_name, "description": "集成测试"})
        if r.status_code == 200:
            kb_id = r.json()["kb_id"]
            print(_bok(f"创建 KB: {kb_id}"))
        else:
            print(_bfail(f"创建失败: {r.status_code}"))
            failures += 1
    except Exception as e:
        print(_bfail(f"异常: {e}"))
        failures += 1

    try:
        r = requests.get(f"{base}/kb/list")
        if r.status_code == 200:
            data = r.json()
            print(_bok(f"KB 列表: {data['total']} 个"))
        else:
            print(_bfail(f"列表失败: {r.status_code}"))
            failures += 1
    except Exception as e:
        print(_bfail(f"异常: {e}"))
        failures += 1

    # 2.3 PDF 上传
    _h1("3. PDF 上传")
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF"
    )
    use_kb = kb_id or "default"
    task_id = None

    try:
        r = requests.post(
            f"{base}/ingest",
            files={"file": ("demo.pdf", io.BytesIO(minimal_pdf), "application/pdf")},
            data={"kb_id": use_kb},
        )
        if r.status_code == 200:
            task_id = r.json()["task_id"]
            print(_bok(f"上传成功: task_id={task_id}"))
        else:
            print(_bwarn(f"上传返回 {r.status_code}: {r.text}"))
    except Exception as e:
        print(_bwarn(f"上传异常: {e}"))

    # 2.4 任务状态
    _h1("4. 任务状态查询")
    if task_id:
        try:
            r = requests.get(f"{base}/ingest/{task_id}")
            if r.status_code == 200:
                data = r.json()
                print(_bok(f"任务 {task_id}: status={data['status']}"))
            else:
                print(_bwarn(f"状态查询返回 {r.status_code}"))
        except Exception as e:
            print(_bfail(f"异常: {e}"))
            failures += 1
    else:
        print(_bwarn("跳过（无 task_id）"))

    # 2.5 RAG 问答
    _h1("5. RAG 问答")
    try:
        r = requests.post(
            f"{base}/query",
            json={
                "question": "multiDal 是什么系统？",
                "kb_ids": [use_kb] if kb_id else [],
                "auto_route": False,
                "rewrite_query": False,
            },
        )
        if r.status_code == 200:
            data = r.json()
            preview = data["answer"][:150]
            print(_bok(f"问答成功: '{preview}...'"))
        else:
            print(_bwarn(f"问答返回 {r.status_code}: {r.text}"))
    except Exception as e:
        print(_bwarn(f"问答异常: {e}"))

    # 2.6 清理
    _h1("6. 清理")
    if kb_id:
        try:
            r = requests.delete(f"{base}/kb/{kb_id}")
            if r.status_code == 200:
                print(_bok(f"已删除 KB: {kb_id}"))
            else:
                print(_bwarn(f"删除返回 {r.status_code}"))
        except Exception as e:
            print(_bfail(f"异常: {e}"))
            failures += 1

    return failures


# ── 主入口 ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="multiDal 集成测试")
    parser.add_argument(
        "--live",
        action="store_true",
        help="对运行中的服务做实时测试（默认使用 TestClient 进程内测试）",
    )
    parser.add_argument("--host", default="http://localhost:8000", help="实时测试的目标地址")
    args = parser.parse_args()

    print(_binfo("multiDal 集成测试"))
    print(f"  模式: {'Live → ' + args.host if args.live else 'TestClient (进程内)'}")

    if args.live:
        failures = _run_live(args.host)
    else:
        failures = _run_testclient()

    print()
    if failures == 0:
        print(_bok("全部测试通过！"))
    else:
        print(_bfail(f"{failures} 项测试失败或异常"))
    print()

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
