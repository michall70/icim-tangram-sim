# -*- coding: utf-8 -*-
"""找到队友的代码，并且**不假设他们的目录叫什么、有几份**。

约定：`third_party/` 下每个子目录都是一份完整的队友仓库（快照），名字随便：

    third_party/20261015/     ← 10 月 15 日那份
    third_party/20261030/     ← 10 月 30 日那份

选哪份，优先级从上到下：

    1. `run.py --team-repo 目录名`（或绝对路径）
    2. `third_party/active.txt` 里写的一行目录名
    3. 只有一份候选 → 就用它
    4. 多份且没指定 → **列出来让你选**，不猜
    5. 一份都没有 → 报错，告诉你怎么放

每份仓库内部长什么样也不假设：按**文件内容**认角色（`class CobotClient` 之类），
见 `discover()`。认错了可以在仓库根目录放 `team_layout.json` 手工指定。
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", "archive", "results", "data", "build",
             "node_modules", ".idea", ".vscode"}

ROLES = ("robot_client", "camera", "calibration", "config", "main",
         "detector", "planner")

ROLE_HINT = {
    "robot_client": "class CobotClient",
    "camera": "class RealSenseCamera",
    "calibration": "class CameraMapper",
    "config": "GRIP_Z + CAMERA_MATRIX",
    "main": "main.py 里的 run(...)",
    "detector": "class TangramDetector",
    "planner": "def plan_puzzle",
}


@dataclass
class TeamLayout:
    root: Path
    modules: dict = field(default_factory=dict)      # 角色 -> 模块名

    def get(self, role: str):
        """按角色取模块对象（找不到返回 None）。"""
        name = self.modules.get(role)
        if not name:
            return None
        return importlib.import_module(name)

    def require(self, role: str):
        mod = self.get(role)
        if mod is None:
            raise SystemExit(
                f"在 {self.root} 里没找到「{role}」（要能找到 "
                f"{ROLE_HINT.get(role, '?')}）。\n"
                f"已经认出来的：{json.dumps(self.modules, ensure_ascii=False, indent=2)}\n"
                f"认错了可以在仓库根目录放 team_layout.json 手工指定，例如：\n"
                f'  {{"robot_client": "common.robot_client", "config": "app.config", '
                f'"main": "app.main"}}')
        return mod

    def score(self) -> int:
        return sum(k in self.modules for k in ROLES)

    def describe(self) -> str:
        lines = [f"    {k:<12} {self.modules[k]}" for k in ROLES if k in self.modules]
        for k in sorted(set(self.modules) - set(ROLES)):
            lines.append(f"    {k:<12} {self.modules[k]}")
        return "\n".join(lines)


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _scan(path: Path):
    """这个 .py 里定义了哪些顶层类 / 函数、有哪些全局赋值。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set(), set(), set()
    classes, funcs, globals_ = set(), set(), set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    globals_.add(t.id)
    return classes, funcs, globals_


def discover(root, *, verbose=False) -> TeamLayout:
    """扫一个目录，按内容认出队友代码里的关键模块。"""
    root = Path(root).resolve()
    found: dict[str, str] = {}
    for py in root.rglob("*.py"):
        if SKIP_DIRS & set(py.parts):
            continue
        classes, funcs, globals_ = _scan(py)
        name = _module_name(root, py)
        if "CobotClient" in classes and "robot_client" not in found:
            found["robot_client"] = name
        if "RealSenseCamera" in classes and "camera" not in found:
            found["camera"] = name
        if "CameraMapper" in classes and "calibration" not in found:
            found["calibration"] = name
        if {"GRIP_Z", "CAMERA_MATRIX"} <= globals_ and "config" not in found:
            found["config"] = name
        if py.stem == "main" and "run" in funcs and "main" not in found:
            found["main"] = name
        if "TangramDetector" in classes and "detector" not in found:
            found["detector"] = name
        if "plan_puzzle" in funcs and "planner" not in found:
            found["planner"] = name

    for cfg in (root / "team_layout.json", root.parent / "team_layout.json"):
        if cfg.is_file():
            try:
                found.update({k: v for k, v in json.loads(
                    cfg.read_text(encoding="utf-8")).items() if isinstance(v, str)})
                if verbose:
                    print(f"[team] 用 {cfg.name} 覆盖了部分布局")
            except Exception as exc:                     # noqa: BLE001
                print(f"[team] 读 {cfg} 失败，忽略：{exc}")

    return TeamLayout(root=root, modules=found)


@dataclass
class Snapshot:
    path: Path
    layout: TeamLayout
    mtime: float

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def score(self) -> int:
        return self.layout.score()


def list_candidates(third_party) -> list:
    """third_party 下所有"像队友仓库"的子目录，按名字排序。"""
    third_party = Path(third_party)
    if not third_party.is_dir():
        return []
    out = []
    for d in sorted(third_party.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        layout = discover(d)
        if layout.score() >= 3:
            out.append(Snapshot(path=d, layout=layout, mtime=d.stat().st_mtime))
    return out


def _active_name(third_party: Path):
    f = Path(third_party) / "active.txt"
    if not f.is_file():
        return None
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def set_active(third_party, name: str) -> Path:
    """把某一份设成默认（写 third_party/active.txt）。"""
    p = Path(third_party) / "active.txt"
    p.write_text(f"{name}\n", encoding="utf-8")
    return p


def format_candidates(third_party) -> str:
    third_party = Path(third_party)
    cands = list_candidates(third_party)
    if not cands:
        return (f"{third_party} 下没有可用的快照"
                f"（要能找到 CobotClient / RealSenseCamera / main.run）")
    active = _active_name(third_party)
    lines = [f"{third_party} 下的候选："]
    for s in cands:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.mtime))
        mark = "*" if s.name == active else " "
        lines.append(f"  {mark} {s.name:<26} 认出 {s.score}/7 项   {when}")
    lines.append("  带 * 的是 third_party/active.txt 指定的默认份")
    return "\n".join(lines)


def resolve_root(third_party, choice=None) -> Path:
    """按 显式指定 → active.txt → 唯一候选 → 报错列清单 的顺序决定用哪份。"""
    third_party = Path(third_party).resolve()
    if choice:
        cand = Path(choice)
        if cand.is_dir():
            return cand.resolve()
        named = third_party / str(choice)
        if named.is_dir():
            return named.resolve()
        raise SystemExit(
            f"--team-repo {choice} 找不到（既不是目录，也不是 {third_party} 下的名字）")

    cands = list_candidates(third_party)
    if not cands:
        raise SystemExit(
            f"{third_party} 下没有可用的队友代码。\n"
            f"把他们的仓库整个拷成 {third_party}/<随便什么名字>/，"
            f"或者用 --team-repo 指定路径。")
    active = _active_name(third_party)
    if active:
        for s in cands:
            if s.name == active:
                return s.path.resolve()
        raise SystemExit(f"third_party/active.txt 写的是 {active!r}，但没有这个目录。\n"
                         + format_candidates(third_party))
    if len(cands) == 1:
        return cands[0].path.resolve()
    raise SystemExit(
        "third_party 下有多份代码，请指定用哪份：\n"
        + format_candidates(third_party)
        + "\n\n用法：python run.py --team-repo <目录名>\n"
          "或者把其中之一写进 third_party/active.txt 当默认。")


def ensure_importable(root: Path) -> None:
    """把仓库根目录放进 sys.path（他们用的是绝对导入）。"""
    r = str(Path(root).resolve())
    if r not in sys.path:
        sys.path.insert(0, r)
