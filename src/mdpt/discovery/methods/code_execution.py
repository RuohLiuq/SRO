"""Shared helpers for code-domain data augmentation generators."""
from __future__ import annotations

import ast
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_DEF_RE = re.compile(r"\bdef\s+([A-Za-z_]\w*)\s*\(")


@dataclass
class CodeVerificationResult:
    passed: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None

    @property
    def feedback(self) -> str:
        parts = []
        if self.error:
            parts.append(self.error)
        if self.stdout.strip():
            parts.append(f"STDOUT:\n{self.stdout.strip()}")
        if self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr.strip()}")
        return "\n\n".join(parts).strip()


def read_json_rows(path: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        obj = json.loads(text)
        rows = [x for x in obj if isinstance(x, dict)]
    else:
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
    return rows[:limit] if limit is not None else rows


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_done_indices(attempts_path: Path, key: str = "input_idx") -> set[int]:
    done: set[int] = set()
    if not attempts_path.exists():
        return done
    with attempts_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key in obj:
                done.add(int(obj[key]))
    return done


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def extract_code(text: str) -> str:
    if not text:
        return ""
    match = _PY_FENCE_RE.search(text)
    if match:
        return match.group(1).strip("\r\n")
    stripped = text.strip("\r\n")
    # LLM outputs are sometimes truncated after an opening fence.  Treat an
    # unclosed leading ```python fence as code instead of feeding backticks to
    # ast.parse / exec.
    lines = stripped.splitlines()
    if lines and re.match(r"^```(?:python|py)?\s*$", lines[0].strip(), flags=re.IGNORECASE):
        return "\n".join(lines[1:]).strip("\r\n")
    return stripped


def fenced_code(code: str) -> str:
    code = extract_code(code)
    return f"```python\n{code}\n```"


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    fenced = _JSON_FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(raw[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


def infer_entry_point(*texts: str) -> Optional[str]:
    for text in texts:
        if not text:
            continue
        match = _DEF_RE.search(text)
        if match:
            return match.group(1)
    return None


def _single_top_level_function(text: str) -> Optional[str]:
    """Infer an entry point only when a program has one unambiguous function."""
    code = extract_code(text)
    if not code.strip():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        match = re.fullmatch(
            r"\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\([^\n]*\)\s*(?:->\s*[^:]+)?\s*:\s*",
            code,
        )
        return match.group(1) if match else None
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "check"
    ]
    return names[0] if len(names) == 1 else None


def function_signature(text: str, entry_point: str) -> Optional[str]:
    """Return a normalized top-level signature for ``entry_point``."""
    code = extract_code(text)
    if not code.strip() or not entry_point:
        return None
    candidates = [code]
    # Some HumanEval-style rows store a header-only prompt. Make that parseable
    # without guessing arguments or other supervision.
    if re.search(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", code):
        candidates.append(code.rstrip() + "\n    pass\n")
    for candidate in candidates:
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
                return f"{prefix} {entry_point}({ast.unparse(node.args)}){returns}:"
    return None


def defines_entry_point(text: str, entry_point: str) -> bool:
    """Whether code explicitly defines the required function at module scope."""
    return function_signature(text, entry_point) is not None


def build_code_task_prompt(question: str, solution: str, entry_point: str) -> str:
    """Build a HumanEval-style prompt that always exposes the exact signature."""
    question = str(question or "").strip()
    entry_point = str(entry_point or "").strip()
    if not entry_point:
        raise ValueError("missing entry_point for HumanEval-style prompt")
    if not re.fullmatch(r"[A-Za-z_]\w*", entry_point):
        raise ValueError("entry_point must be a bare function name")
    if function_signature(question, entry_point):
        try:
            ast.parse(extract_code(question))
        except SyntaxError:
            return f"{question.rstrip()}\n    pass"
        return question
    signature = function_signature(solution, entry_point)
    if not signature:
        raise ValueError(f"cannot recover signature for entry_point={entry_point!r}")
    description = question.replace('"""', '\\\"\\\"\\\"').strip()
    if not description:
        return signature
    docstring = "\n".join(f"    {line}" for line in description.splitlines())
    return f'{signature}\n    """\n{docstring}\n    """'


def code_verifier_fields(row: dict[str, Any]) -> dict[str, str]:
    """Extract code-verifier metadata (tests + entry_point) from a raw row."""
    tests = row.get("tests") or row.get("test") or ""
    entry_point = row.get("entry_point") or ""
    if not entry_point:
        prompt = first_present(row, ("prompt", "question", "problem", "instruction", "text"))
        solution = first_present(
            row,
            ("canonical_solution", "solution", "answer", "code", "response", "output"),
        )
        entry_point = (
            _single_top_level_function(str(prompt or ""))
            or _single_top_level_function(str(solution or ""))
            or ""
        )
    return {"tests": str(tests or ""), "entry_point": str(entry_point or "")}


def _extract_import_lines(text: str) -> list[str]:
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(stripped)
    return lines


def _prepend_missing_imports(code: str, prompt: str) -> str:
    imports = [line for line in _extract_import_lines(prompt) if line not in code]
    if not imports:
        return code
    return "\n".join(imports) + "\n\n" + code


def _as_function_body(code: str) -> str:
    body = code.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4).strip("\n")
    if not body:
        return ""
    first = next((line for line in body.splitlines() if line.strip()), "")
    leading = len(first) - len(first.lstrip(" "))
    # HumanEval-style datasets already store body continuations with their
    # required indentation. Preserve them byte-for-byte apart from newline/tab
    # normalization: global dedent corrupts triple-quoted string contents whose
    # internal lines intentionally start in column zero.
    if leading >= 4:
        return body
    return textwrap.indent(body, "    ")


def build_solution_program(solution: str, *, prompt: str = "", entry_point: str = "") -> str:
    code = extract_code(solution)
    if entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code):
        return _prepend_missing_imports(code.strip("\r\n"), prompt)
    if prompt and "def " in prompt:
        return f"{prompt.rstrip()}\n{_as_function_body(code)}\n"
    return _prepend_missing_imports(code.strip("\r\n"), prompt)


def humaneval_sft_completion(solution: str, *, prompt: str, entry_point: str) -> str:
    """Return a complete, unfenced program for scorer-backed HumanEval SFT.

    Inspect's HumanEval scorer executes unfenced completions verbatim after the
    benchmark prompt. A complete program is robust to generation-time loss of
    leading indentation and preserves module-level helpers in their original
    order. Body-only sources are reconstructed by ``build_solution_program``.
    """
    entry_point = str(entry_point or "").strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", entry_point):
        raise ValueError("entry_point must be a bare function name")

    normalized_solution = extract_code(solution).replace("\r\n", "\n").replace("\r", "\n")
    normalized_prompt = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n")
    program = build_solution_program(
        normalized_solution,
        prompt=normalized_prompt,
        entry_point=entry_point,
    ).expandtabs(4)
    try:
        tree = ast.parse(program)
    except SyntaxError as exc:
        raise ValueError(f"cannot parse solution program for {entry_point!r}: {exc}") from exc

    targets = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
    ]
    if len(targets) != 1:
        raise ValueError(
            f"solution program must define exactly one top-level function {entry_point!r}"
        )
    completion = program.strip("\r\n")
    if "\r" in completion or "\t" in completion or completion.startswith("```"):
        raise AssertionError("HumanEval SFT completion normalization failed")
    return completion.rstrip()


def prepare_humaneval_sft(
    *,
    question: str,
    solution: str,
    entry_point: str,
) -> tuple[str, str, str]:
    """Build one canonical prompt/completion/program tuple for every code path."""
    solution_program = build_solution_program(
        extract_code(solution),
        prompt=question,
        entry_point=entry_point,
    )
    code_prompt = build_code_task_prompt(question, solution_program, entry_point)
    completion = humaneval_sft_completion(
        solution_program,
        prompt=code_prompt,
        entry_point=entry_point,
    )
    return code_prompt, completion, solution_program


def official_humaneval_answer(completion: str) -> str:
    """Apply the exact extractor used by the pinned Inspect HumanEval scorer."""
    from inspect_evals.humaneval.humaneval import find_code

    return str(find_code(str(completion or "")))


def verify_humaneval_sft_completion(
    *,
    prompt: str,
    completion: str,
    tests: str,
    entry_point: str,
    timeout_s: float = 5.0,
) -> CodeVerificationResult:
    """Execute the exact pinned Inspect HumanEval scorer path."""
    value = str(completion or "")
    if not value.strip():
        return CodeVerificationResult(False, error="empty HumanEval completion")
    answer = official_humaneval_answer(value)
    prompt_text = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n").rstrip()
    scorer_program = f"{prompt_text}\n{answer}\n"
    return verify_python_solution(
        prompt="",
        solution=scorer_program,
        tests=tests,
        entry_point=entry_point,
        timeout_s=timeout_s,
    )


def verify_python_solution(
    *,
    prompt: str,
    solution: str,
    tests: str,
    entry_point: str,
    timeout_s: float = 5.0,
    require_explicit_entry_point: bool = False,
) -> CodeVerificationResult:
    if require_explicit_entry_point and not defines_entry_point(solution, entry_point):
        return CodeVerificationResult(
            False,
            error=f"solution must define the required top-level function {entry_point!r}",
        )
    solution_program = build_solution_program(solution, prompt=prompt, entry_point=entry_point)
    test_code = tests.strip()
    if not solution_program.strip():
        return CodeVerificationResult(False, error="empty solution")
    if not test_code:
        return CodeVerificationResult(False, error="empty tests")
    if not entry_point:
        return CodeVerificationResult(False, error="missing entry_point")
    if "def check" not in test_code:
        return CodeVerificationResult(False, error="tests must define check(candidate)")
    try:
        ast.parse(solution_program)
        ast.parse(test_code)
    except SyntaxError as exc:
        return CodeVerificationResult(False, error=f"SyntaxError: {exc}")
    if not defines_entry_point(solution_program, entry_point):
        return CodeVerificationResult(
            False,
            error=f"solution program does not define entry_point {entry_point!r}",
        )

    cpu_seconds = max(1, int(math.ceil(timeout_s)))
    resource_bootstrap = (
        "import resource\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        "resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))\n"
        "resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))\n"
    )
    program = f"{resource_bootstrap}\n{solution_program}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "verify_candidate.py"
        stdout_path = Path(tmp) / "stdout.txt"
        stderr_path = Path(tmp) / "stderr.txt"
        path.write_text(program, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        with stdout_path.open("w+", encoding="utf-8") as stdout_file, stderr_path.open("w+", encoding="utf-8") as stderr_file:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(path)],
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=tmp,
                env=env,
                start_new_session=True,
                text=True,
            )
            timed_out = False
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(65536)
            stderr = stderr_file.read(65536)
        if timed_out:
            return CodeVerificationResult(False, stdout=stdout, stderr=stderr, error="TimeoutExpired")
    return CodeVerificationResult(
        proc.returncode == 0,
        stdout=stdout,
        stderr=stderr,
        error=None if proc.returncode == 0 else f"returncode={proc.returncode}",
    )
