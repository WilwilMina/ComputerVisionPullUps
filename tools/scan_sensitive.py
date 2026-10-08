"""Scan for sensitive content before it is committed or pushed. Standard library only; no network.

    python tools/scan_sensitive.py --staged            # what is about to be committed (pre-commit hook)
    python tools/scan_sensitive.py --push-stdin        # commits about to be pushed (pre-push hook, reads git's stdin)
    python tools/scan_sensitive.py --range A..B        # an explicit revision range
    python tools/scan_sensitive.py --tracked           # every file in HEAD

It blocks: videos, model files, keypoint caches, PDFs, archives, keys and .env files; anything under data/,
samples/, out/ or models/; files over a size limit; private keys and common API-token shapes; assignments of
long literals to secret-looking names; personal absolute paths and email addresses; and commits authored with an
email that is not a GitHub no-reply address. Findings never print the secret itself, only a redacted hint.

Limits (a regex scanner cannot do better): encoded or split-up secrets (base64, "ghp_" + "..."), secrets in
commit messages, and unquoted letters-only values. A line containing the marker `scan:allow` is skipped. Do not
bypass the hooks with --no-verify.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Iterator, Optional, Sequence

MAX_FILE_BYTES = 1_000_000
MAX_SCAN_CHARS = 20_000  # longer lines are scanned in overlapping chunks so scan time stays linear
CHUNK_OVERLAP = 400
ALLOW_MARKER = "scan:allow"

BLOCKED_DIRS = ("data/", "samples/", "out/", "models/", ".venv/", "__pycache__/", ".pytest_cache/")
BLOCKED_SUFFIXES = (
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",  # videos
    ".npz", ".npy", ".pkl", ".pickle",  # keypoint caches and pickles
    ".task", ".pt", ".pth", ".onnx", ".ckpt", ".safetensors", ".h5",  # model files
    ".pdf", ".zip", ".7z", ".tar", ".gz", ".rar",  # documents and archives
    ".pem", ".key", ".p12", ".pfx", ".jks",  # key material
)
BLOCKED_NAMES = ("id_rsa", "id_ed25519", "settings.local.json", ".netrc", ".npmrc", ".pypirc")
ENV_TEMPLATE_NAMES = (".env.example", ".env.sample", ".env.template")  # placeholders by convention; content is still scanned

# Only a GitHub no-reply address may author a commit that gets pushed.
ALLOWED_AUTHOR_SUFFIX = "@users.noreply.github.com"
# Email addresses that are fine to appear in files (placeholders, no-reply addresses, the git@ clone syntax).
ALLOWED_EMAIL_DOMAINS = ("users.noreply.github.com", "github.com", "anthropic.com", "example.com", "example.org", "example.net")

_NAME = r"(?:api[_-]?key|secret|token|passwd|password|auth)\w*"
SECRET_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY(?: BLOCK)?-----")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9_\-]{32,}")),  # includes sk-proj-... and sk-svcacct-...
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("JSON web token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    # name = "literal", name: 'literal', and JSON {"name": "literal"}
    ("secret-looking assignment", re.compile(r"""(?i)\b""" + _NAME + r"""["']?\s*[:=]\s*['"][^'"\s]{12,}['"]""")),
    # unquoted dotenv/YAML values: long, and mixing letters with digits so ordinary words are not flagged
    (
        "secret-looking assignment",
        re.compile(r"(?i)\b" + _NAME + r"""["']?\s*[:=]\s*(?=[A-Za-z0-9_\-/+=]*\d)(?=[A-Za-z0-9_\-/+=]*[A-Za-z])[A-Za-z0-9_\-/+=]{16,}"""),
    ),
)
PATH_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("personal Windows path", re.compile(r"(?i)\b[a-z]:[\\/]+users[\\/]+(?!<)[\w.\-]+")),  # scan:allow
    ("personal macOS/Linux path", re.compile(r"(?:/Users|/home)/(?!<|\.\.)[\w.\-]+/")),  # scan:allow
)
# Anchored on the left (lookbehind) so a long run of word characters with no '@' cannot cause quadratic scanning.
EMAIL_PATTERN = re.compile(r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]{1,64}@([A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,})")


@dataclass(frozen=True)
class Finding:
    where: str  # path, or commit id for commit-level findings
    line: Optional[int]
    kind: str
    hint: str  # redacted; never the full secret

    def __str__(self) -> str:
        loc = self.where if self.line is None else f"{self.where}:{self.line}"
        return f"  {loc}: {self.kind} ({self.hint})"


def _redact(text: str) -> str:
    text = text.strip()
    return f"{text[:4]}... ({len(text)} chars)" if len(text) > 8 else "..."


def _redact_email(email: str) -> str:
    return "...@" + email.rsplit("@", 1)[-1]  # the local part can be a name; show the domain only


def scan_path(path: str, size: int) -> list[Finding]:
    """Findings that depend only on the file's name and size."""
    p = PurePosixPath(path.replace("\\", "/"))
    posix = p.as_posix()
    lower_name, lower_suffix = p.name.lower(), p.suffix.lower()
    out: list[Finding] = []
    if any(posix.startswith(d) or f"/{d}" in f"/{posix}" for d in BLOCKED_DIRS):
        out.append(Finding(posix, None, "blocked directory", "data, samples, outputs, models and caches are never committed"))
    if lower_suffix in BLOCKED_SUFFIXES:
        out.append(Finding(posix, None, "blocked file type", lower_suffix))
    is_env = lower_name == ".env" or (lower_name.startswith(".env.") and lower_name not in ENV_TEMPLATE_NAMES)
    if lower_name in BLOCKED_NAMES or is_env:
        out.append(Finding(posix, None, "blocked file name", p.name))
    if size > MAX_FILE_BYTES:
        out.append(Finding(posix, None, "file too large", f"{size} bytes > {MAX_FILE_BYTES}"))
    return out


def _chunks(line: str) -> Iterator[str]:
    if len(line) <= MAX_SCAN_CHARS:
        yield line
        return
    step = MAX_SCAN_CHARS - CHUNK_OVERLAP
    for start in range(0, len(line), step):
        yield line[start : start + MAX_SCAN_CHARS]


def scan_text(path: str, text: str) -> list[Finding]:
    """Findings inside a text file's content."""
    out: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        seen: set[tuple[str, str]] = set()
        for chunk in _chunks(line):
            for kind, pattern in SECRET_PATTERNS + PATH_PATTERNS:
                m = pattern.search(chunk)
                if m and (kind, m.group(0)) not in seen:
                    seen.add((kind, m.group(0)))
                    out.append(Finding(path, number, kind, _redact(m.group(0))))
            if "@" in chunk:
                for m in EMAIL_PATTERN.finditer(chunk):
                    domain = m.group(1).lower()
                    if not any(domain == d or domain.endswith("." + d) for d in ALLOWED_EMAIL_DOMAINS):
                        if ("email address", m.group(0)) not in seen:
                            seen.add(("email address", m.group(0)))
                            out.append(Finding(path, number, "email address", _redact_email(m.group(0))))
    return out


def scan_blob(path: str, data: bytes) -> list[Finding]:
    findings = scan_path(path, len(data))
    if len(data) > MAX_FILE_BYTES:
        return findings  # already blocked by size; do not spend time reading it
    if b"\x00" in data[:8192]:
        findings.append(Finding(path.replace("\\", "/"), None, "binary file", "only text files are committed"))
        return findings
    findings.extend(scan_text(path.replace("\\", "/"), data.decode("utf-8", errors="replace")))
    return findings


def scan_author(commit: str, author_email: str, committer_email: str) -> list[Finding]:
    out = []
    for role, email in (("author", author_email), ("committer", committer_email)):
        if not email.lower().endswith(ALLOWED_AUTHOR_SUFFIX):
            out.append(Finding(commit[:10], None, f"commit {role} email is not a GitHub no-reply address", _redact_email(email)))
    return out


# --- git plumbing -----------------------------------------------------------------------------------------

def _git(*args: str, check: bool = True) -> bytes:
    res = subprocess.run(["git", *args], capture_output=True)
    if check and res.returncode != 0:
        raise SystemExit(f"scan_sensitive: git {' '.join(args)} failed: {res.stderr.decode(errors='replace').strip()}")
    return res.stdout


def _names(output: bytes) -> list[str]:
    return [n.decode("utf-8", errors="replace") for n in output.split(b"\0") if n]


def _names_lines(output: bytes) -> list[str]:
    return [l.strip() for l in output.decode().splitlines() if l.strip()]


def scan_staged() -> list[Finding]:
    findings: list[Finding] = []
    for path in _names(_git("diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z")):
        findings.extend(scan_blob(path, _git("show", f":{path}")))
    return findings


def scan_commit(commit: str, check_author: bool = True) -> list[Finding]:
    findings: list[Finding] = []
    if check_author:
        ident = _git("log", "-1", "--format=%ae%n%ce", commit).decode().split("\n")
        findings.extend(scan_author(commit, ident[0].strip(), ident[1].strip()))
    # -m: for a merge commit, diff against each parent so content that exists only in the merge is still scanned
    listing = _git("diff-tree", "-m", "--root", "-r", "--no-commit-id", "--name-only", "--diff-filter=ACMRT", "-z", commit)
    for path in dict.fromkeys(_names(listing)):
        findings.extend(scan_blob(path, _git("show", f"{commit}:{path}")))
    return findings


def scan_commits(commits: Iterable[str]) -> list[Finding]:
    seen: set[Finding] = set()
    findings: list[Finding] = []
    for c in commits:
        for f in scan_commit(c):
            if f not in seen:
                seen.add(f)
                findings.append(f)
    return findings


def scan_tracked() -> list[Finding]:
    findings: list[Finding] = []
    for path in _names(_git("ls-tree", "-r", "--name-only", "-z", "HEAD")):
        findings.extend(scan_blob(path, _git("show", f"HEAD:{path}")))
    return findings


def commits_from_push_stdin(stdin_lines: Iterable[str]) -> list[str]:
    """Commits a push would send, from git's pre-push stdin: '<local ref> <local sha> <remote ref> <remote sha>'."""
    zero = "0" * 40
    commits: list[str] = []
    for line in stdin_lines:
        parts = line.split()
        if len(parts) < 4 or parts[1] == zero:
            continue  # deleting a ref
        local_sha, remote_sha = parts[1], parts[3]
        rev_args = [local_sha, "--not", "--remotes"] if remote_sha == zero else [f"{remote_sha}..{local_sha}"]
        commits.extend(_names_lines(_git("rev-list", *rev_args)))
    return list(dict.fromkeys(commits))


def report(findings: Sequence[Finding], what: str) -> int:
    if not findings:
        print(f"scan_sensitive: OK ({what})")
        return 0
    print(f"scan_sensitive: BLOCKED ({what}): {len(findings)} finding(s)", file=sys.stderr)
    for f in findings:
        print(str(f), file=sys.stderr)
    print(
        "Remove or unstage the item (or, for a false positive on one line, add the marker 'scan:allow' to that line). "
        "Do not bypass with --no-verify.",
        file=sys.stderr,
    )
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Scan for sensitive content before commit or push.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true", help="scan the staged changes")
    group.add_argument("--push-stdin", action="store_true", help="scan the commits a push would send (git pre-push stdin)")
    group.add_argument("--range", help="scan the commits in a revision range, e.g. origin/main..HEAD")
    group.add_argument("--tracked", action="store_true", help="scan every file in HEAD")
    args, _extra = ap.parse_known_args(argv)  # the pre-push hook also passes the remote name and URL

    if args.staged:
        return report(scan_staged(), "staged changes")
    if args.push_stdin:
        commits = commits_from_push_stdin(sys.stdin)
        return report(scan_commits(commits), f"{len(commits)} commit(s) to push")
    if args.range:
        commits = _names_lines(_git("rev-list", args.range))
        return report(scan_commits(commits), f"{len(commits)} commit(s) in {args.range}")
    return report(scan_tracked(), "tracked files at HEAD")


if __name__ == "__main__":
    sys.exit(main())
