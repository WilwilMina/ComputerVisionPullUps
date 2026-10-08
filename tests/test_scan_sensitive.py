"""The scanner's own tests. Fake secrets are assembled at runtime so this file does not trip the scanner itself."""
import subprocess

import pytest

from tools import scan_sensitive as sc


def kinds(findings):
    return {f.kind for f in findings}


# ---------- file names, folders, size ----------

@pytest.mark.parametrize(
    "path",
    ["data/external/a/clip.txt", "samples/x.txt", "out/render.txt", "models/readme.txt", "a/b/.venv/lib.py", "x/__pycache__/m.py"],
)
def test_blocked_directories(path):
    assert "blocked directory" in kinds(sc.scan_path(path, 10))


@pytest.mark.parametrize(
    "path",
    ["clip.mp4", "a/b/clip.WEBM", "kp.npz", "pose_landmarker_heavy.task", "w.onnx", "guide.pdf", "archive.zip", "id.pem", "k.key"],
)
def test_blocked_file_types(path):
    assert "blocked file type" in kinds(sc.scan_path(path, 10))


@pytest.mark.parametrize("path", [".env", "cfg/.env", ".env.local", ".claude/settings.local.json", "id_rsa", ".netrc"])
def test_blocked_file_names(path):
    assert "blocked file name" in kinds(sc.scan_path(path, 10))


def test_large_files_are_blocked_but_the_limit_itself_is_fine():
    assert "file too large" in kinds(sc.scan_path("a.txt", sc.MAX_FILE_BYTES + 1))
    assert sc.scan_path("a.txt", sc.MAX_FILE_BYTES) == []


@pytest.mark.parametrize("path", ["pullup_rom/pose.py", "tests/test_pose.py", "docs/PROGRESS.md", ".claude/settings.json", "CLAUDE.md"])
def test_normal_project_files_pass(path):
    assert sc.scan_path(path, 5000) == []


# ---------- secrets in content ----------

def fake_secrets():
    return {
        "private key": "-----BEGIN " + "RSA PRIVATE KEY-----",
        "GitHub token": "ghp_" + "a" * 36,
        "Anthropic API key": "sk-ant-" + "b" * 30,
        "AWS access key id": "AKIA" + "ABCDEFGHIJKLMNOP",
        "Google API key": "AIza" + "c" * 35,
        "Slack token": "xoxb-" + "1234567890-abcdef",
        "JSON web token": "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12,
        "OpenAI-style API key": "sk-" + "d" * 40,
    }


@pytest.mark.parametrize("kind", list(fake_secrets()))
def test_each_secret_shape_is_detected_and_redacted(kind):
    secret = fake_secrets()[kind]
    findings = sc.scan_text("a.py", f"x = 1\nvalue = '{secret}'\n")
    hit = [f for f in findings if f.kind == kind]
    assert hit and hit[0].line == 2
    assert secret not in str(hit[0])  # the report never repeats the secret


def test_secret_looking_assignment_is_detected_but_short_values_are_not():
    long_value = "x" * 20
    assert "secret-looking assignment" in kinds(sc.scan_text("a.py", "api_key = '" + long_value + "'"))
    assert "secret-looking assignment" in kinds(sc.scan_text("a.py", 'PASSWORD: "' + long_value + '"'))
    assert sc.scan_text("a.py", "api_key = 'short'") == []
    assert sc.scan_text("a.py", "token_count = 12345678901234") == []  # no quoted literal


def test_allow_marker_skips_only_that_line():
    secret = "ghp_" + "a" * 36
    text = f"a = '{secret}'  # scan:allow\nb = '{secret}'\n"
    findings = sc.scan_text("a.py", text)
    assert [f.line for f in findings] == [2]


# ---------- personal paths and emails ----------

def test_personal_paths_are_flagged():
    windows = "C:" + "\\Users\\" + "someone" + "\\Documents\\x"
    unix = "/home/" + "someone" + "/project"
    mac = "/Users/" + "someone" + "/project"
    for text in (windows, windows.replace("\\", "/"), unix, mac):
        assert kinds(sc.scan_text("d.md", text)) & {"personal Windows path", "personal macOS/Linux path"}, text


def test_placeholder_and_relative_paths_are_not_flagged():
    assert sc.scan_text("d.md", "C:" + "\\Users\\<you>\\Documents") == []
    assert sc.scan_text("d.md", "see ../data/clip.mp4 and docs/PROGRESS.md") == []


def test_emails_only_no_reply_and_placeholder_domains_are_allowed():
    ok = ["Co-Authored-By: Claude <noreply@" + "anthropic.com>", "183628245+someone@users." + "noreply.github.com", "a@" + "example.com"]
    for text in ok:
        assert sc.scan_text("d.md", text) == [], text
    for domain in ("gmail.com", "ncsu.edu", "outlook.com"):
        findings = sc.scan_text("d.md", "me@" + domain)
        assert kinds(findings) == {"email address"}
        assert "me@" + domain not in str(findings[0])


def test_commit_authors_must_be_github_no_reply_addresses():
    good = "183628245+someone@users." + "noreply.github.com"
    assert sc.scan_author("abc1234567890", good, good) == []
    bad = sc.scan_author("abc1234567890", "me@" + "ncsu.edu", good)
    assert len(bad) == 1 and "author" in bad[0].kind
    both = sc.scan_author("abc1234567890", "x@" + "gmail.com", "y@" + "gmail.com")
    assert len(both) == 2


def test_binary_blobs_are_blocked():
    assert "binary file" in kinds(sc.scan_blob("thing.dat", b"abc\x00def"))
    assert sc.scan_blob("notes.md", b"plain text only\n") == []


# ---------- git integration, in a throwaway repo ----------

def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Tester")
    git(tmp_path, "config", "user.email", "1+tester@users." + "noreply.github.com")
    git(tmp_path, "config", "core.hooksPath", "none")  # do not run this project's hooks inside the temp repo
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.slow
def test_staged_scan_blocks_a_secret_and_passes_clean_files(repo, capsys):
    (repo / "ok.py").write_text("x = 1\n")
    git(repo, "add", "ok.py")
    assert sc.main(["--staged"]) == 0

    (repo / "bad.py").write_text("k = '" + "ghp_" + "a" * 36 + "'\n")
    (repo / "clip.mp4").write_bytes(b"\x00\x00\x00\x20ftypmp42")
    git(repo, "add", "bad.py", "clip.mp4")
    assert sc.main(["--staged"]) == 1
    err = capsys.readouterr().err
    assert "GitHub token" in err and "blocked file type" in err and "ghp_" + "a" * 36 not in err


@pytest.mark.slow
def test_push_scan_checks_every_commit_and_every_author_not_just_the_last_one(repo):
    (repo / "leak.py").write_text("k = '" + "AKIA" + "ABCDEFGHIJKLMNOP" + "'\n")
    git(repo, "add", "leak.py")
    git(repo, "-c", "user.email=me@" + "ncsu.edu", "commit", "-q", "-m", "oops")  # personal author email too
    (repo / "leak.py").write_text("k = 'removed'\n")  # fixed in a later commit, but the secret is still in history
    git(repo, "commit", "-q", "-am", "fix")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    commits = sc.commits_from_push_stdin([f"refs/heads/main {head} refs/heads/main {'0' * 40}\n"])
    assert len(commits) == 2
    found = sc.scan_commits(commits)
    assert "AWS access key id" in kinds(found)
    assert any("author" in f.kind for f in found)
    assert not any(f.where == commits[0][:10] for f in found)  # the clean no-reply commit has no author finding


def test_deleting_a_remote_ref_scans_nothing():
    assert sc.commits_from_push_stdin(["(delete) " + "0" * 40 + " refs/heads/x " + "a" * 40 + "\n"]) == []


# ---------- regressions from the QA review ----------

def test_a_very_long_line_is_scanned_in_linear_time():
    import time

    start = time.time()
    sc.scan_text("min.js", "a" * 100_000)  # used to take about a minute: quadratic email regex
    sc.scan_text("min.js", "word " * 20_000)
    assert time.time() - start < 2.0


def test_a_secret_hidden_deep_inside_a_huge_line_is_still_found():
    token = "ghp_" + "a" * 36
    line = "x" * 30_000 + " " + token + " " + "y" * 30_000  # beyond one chunk, so chunking must not lose it
    assert "GitHub token" in kinds(sc.scan_text("min.js", line))
    boundary = "z" * (sc.MAX_SCAN_CHARS - 10) + " " + token  # straddles a chunk edge
    assert "GitHub token" in kinds(sc.scan_text("min.js", boundary))


def test_oversized_files_are_flagged_without_reading_their_content():
    findings = sc.scan_blob("big.txt", b"a" * (sc.MAX_FILE_BYTES + 1))
    assert kinds(findings) == {"file too large"}


def test_modern_openai_project_keys_are_detected():
    assert "OpenAI-style API key" in kinds(sc.scan_text("a.txt", "sk-proj-" + "A" * 40))
    assert "OpenAI-style API key" in kinds(sc.scan_text("a.txt", "sk-svcacct-" + "B" * 40))


def test_json_yaml_and_dotenv_style_secrets_are_detected():
    long_value = "abcdefghijklmnop"
    assert "secret-looking assignment" in kinds(sc.scan_text("c.json", '{"api_key": "' + long_value + '"}'))
    assert "secret-looking assignment" in kinds(sc.scan_text("c.yaml", "password=" + "hunter2" * 3))
    assert "secret-looking assignment" in kinds(sc.scan_text("c.yaml", "token: " + "abc123def456ghi789jkl"))
    # ordinary code and words are not flagged
    assert sc.scan_text("c.py", "token_count = compute_tokens(text)") == []
    assert sc.scan_text("c.md", "authentication_required is a status word") == []


def test_blocked_names_are_case_insensitive_and_env_templates_are_allowed():
    for name in (".ENV", ".Env.local", ".claude/Settings.Local.json", "ID_RSA"):
        assert "blocked file name" in kinds(sc.scan_path(name, 10)), name
    for name in (".env.example", ".env.sample", ".env.template"):
        assert sc.scan_path(name, 10) == [], name
    assert "AKIA" + "ABCDEFGHIJKLMNOP" in "AKIA" + "ABCDEFGHIJKLMNOP"  # guard against a typo in the fixture above
    assert "AWS access key id" in kinds(sc.scan_text(".env.example", "KEY=" + "AKIA" + "ABCDEFGHIJKLMNOP"))  # templates are still content-scanned


def test_email_findings_show_only_the_domain():
    findings = sc.scan_text("d.md", "contact: jane.doe@" + "gmail.com")
    assert findings and "jane" not in str(findings[0]) and "gmail.com" in str(findings[0])
    assert sc.scan_text("d.md", "clone with git@" + "github.com:user/repo.git") == []


@pytest.mark.slow
def test_merge_commit_content_is_scanned_on_push(repo):
    (repo / "a.txt").write_text("base")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "a.txt").write_text("side change")
    git(repo, "commit", "-q", "-am", "side")
    git(repo, "checkout", "-q", "main")
    (repo / "a.txt").write_text("main change")
    git(repo, "commit", "-q", "-am", "main")
    merge = subprocess.run(["git", "merge", "side"], cwd=repo, capture_output=True)
    assert merge.returncode != 0  # conflict
    (repo / "a.txt").write_text("k = '" + "AKIA" + "ABCDEFGHIJKLMNOP" + "'")  # the secret exists only in the resolution
    git(repo, "add", "a.txt")
    git(repo, "commit", "-q", "-m", "merge")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert "AWS access key id" in kinds(sc.scan_commit(head))
