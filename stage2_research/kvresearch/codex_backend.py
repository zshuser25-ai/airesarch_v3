from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


MAIN_AGENT_MODEL = "gpt-6.1-sol"
MAIN_AGENT_REASONING_EFFORT = "xhigh"
MAIN_AGENT_SERVICE_TIER = "fast"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-subagents", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_subagents < 1 or args.max_subagents > 16:
        parser.error("--max-subagents must be between 1 and 16")
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    project_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) + ((":" + existing) if existing else "")
    command = [
        "codex", "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--model", MAIN_AGENT_MODEL,
        "-c", f'model_reasoning_effort="{MAIN_AGENT_REASONING_EFFORT}"',
        "-c", f'service_tier="{MAIN_AGENT_SERVICE_TIER}"',
        "-c", "features.fast_mode=true",
        "-c", 'approval_policy="never"',
        "-c", 'web_search="live"',
        "-c", "sandbox_workspace_write.network_access=true",
        "-c", "agents.enabled=true",
        "-c", f'agents.default_subagent_model="{MAIN_AGENT_MODEL}"',
        "-c", f'agents.default_subagent_reasoning_effort="{MAIN_AGENT_REASONING_EFFORT}"',
        "-c", f"agents.max_concurrent_threads_per_session={args.max_subagents}",
        "--sandbox", "workspace-write",
        "--cd", str(Path(args.run).resolve()),
        prompt,
    ]
    result = subprocess.run(command, env=env, text=True, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
