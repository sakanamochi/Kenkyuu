"""Blender MCPへ標準stdio接続する最小クライアント。"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


RUNTIME_DIR = Path(__file__).resolve().parent
SERVER_EXE = RUNTIME_DIR / "bin" / "blender-mcp.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blender MCPツールを一覧・実行する")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="利用可能なMCPツールを一覧表示する")
    subparsers.add_parser("scene", help="現在のBlenderシーン情報を取得する")
    object_parser = subparsers.add_parser(
        "object",
        help="指定オブジェクトの詳細情報を取得する",
    )
    object_parser.add_argument("object_name")

    call_parser = subparsers.add_parser("call", help="MCPツールを1回実行する")
    call_parser.add_argument("tool_name")
    call_parser.add_argument(
        "arguments_json",
        nargs="?",
        default="{}",
        help="ツール引数のJSONオブジェクト",
    )

    execute_parser = subparsers.add_parser(
        "execute",
        help="UTF-8のPythonファイルをBlender内で実行する",
    )
    execute_parser.add_argument("python_file", type=Path)
    execute_parser.add_argument("--prompt", default="Blenderシーンを段階的に編集する")

    screenshot_parser = subparsers.add_parser(
        "screenshot",
        help="3Dビューポート画像をPNGとして保存する",
    )
    screenshot_parser.add_argument("output_file", type=Path)
    screenshot_parser.add_argument("--max-size", type=int, default=1200)
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    """PydanticモデルをJSON化可能な値へ変換する。"""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


async def main() -> None:
    args = parse_args()
    server = StdioServerParameters(
        command=str(SERVER_EXE),
        args=[],
        env={
            **os.environ,
            "BLENDER_HOST": "localhost",
            "BLENDER_PORT": "9876",
            "DISABLE_TELEMETRY": "true",
        },
    )

    async with stdio_client(server) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            if args.command == "list":
                result = await session.list_tools()
            elif args.command == "scene":
                result = await session.call_tool(
                    "get_scene_info",
                    {
                        "user_prompt": (
                            "元の1194M_3モデルから高品質な比較用PAFモデルを"
                            "一から制作するために現在のシーンを確認する"
                        )
                    },
                )
            elif args.command == "object":
                result = await session.call_tool(
                    "get_object_info",
                    {
                        "object_name": args.object_name,
                        "user_prompt": "元モデルの寸法と軸方向を正確に確認する",
                    },
                )
            elif args.command == "execute":
                code = args.python_file.resolve().read_text(encoding="utf-8")
                result = await session.call_tool(
                    "execute_blender_code",
                    {"code": code, "user_prompt": args.prompt},
                )
            elif args.command == "screenshot":
                result = await session.call_tool(
                    "get_viewport_screenshot",
                    {
                        "max_size": args.max_size,
                        "user_prompt": "制作中モデルの形状と品質を目視確認する",
                    },
                )
                image_items = [
                    item
                    for item in result.content
                    if getattr(item, "type", None) == "image"
                ]
                if not image_items:
                    raise RuntimeError("MCP応答にビューポート画像が含まれていません")
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                args.output_file.write_bytes(base64.b64decode(image_items[0].data))
                result = {
                    "saved": str(args.output_file.resolve()),
                    "mime_type": image_items[0].mimeType,
                }
            else:
                arguments = json.loads(args.arguments_json)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments_jsonはJSONオブジェクトで指定してください")
                result = await session.call_tool(args.tool_name, arguments)

    print(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
