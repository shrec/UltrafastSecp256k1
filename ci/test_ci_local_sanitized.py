#!/usr/bin/env python3
# ============================================================================
# test_ci_local_sanitized.py — deterministic regression suite for the hardened
# local/pre-push CI entrypoint ci/ci_local_sanitized.sh.
#
# The wrapper reduces the process to a minimal, trusted environment, drops
# inherited shell-startup / function and Git routing / config state, and then
# delegates the caller's UNCHANGED arguments to the tracked sibling
# ci/ci_local.sh. These tests prove each of those properties with real
# sub-processes and adversarial (hostile) inputs.
#
# Portability: repositories are laid out from plain files (never `git init`)
# and inspected with READ-ONLY git only, so the suite runs identically on hosts
# whose temp filesystem forbids chmod/utime and on hosts where a global/system
# `safe.bareRepository=explicit` policy is active. Every git invocation that
# inspects a repository is host-policy-independent (explicit --git-dir /
# --file plus GIT_CONFIG_GLOBAL/SYSTEM neutralization).
#
# Run:  python3 ci/test_ci_local_sanitized.py
# ============================================================================
from __future__ import annotations

import hashlib
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "ci_local_sanitized.sh"
TESTFILE = HERE / "test_ci_local_sanitized.py"
SIBLING = HERE / "ci_local.sh"
REQUIRED = (WRAPPER, TESTFILE)

# The ONLY supported security entrypoint.
ENTRY = ("/bin/bash", "-p")

# Bash auto-exports these under `env -i`; the wrapper additionally establishes
# the trusted Git-config neutralizers. Nothing else may reach ci/ci_local.sh.
ALLOWED_ENV_NAMES = {
    "PATH", "HOME", "PWD", "SHLVL", "OLDPWD", "_",
    "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM",
}
SEAL_NAME = "__UFCI_SANITIZED_V3"

# Read-only git that ignores host global/system configuration.
GIT_NEUTRAL = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/nonexistent",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}

# Common preamble for every stub: resolve the sandbox root (parent of ci/)
# from the stub's own location, independent of cwd and of any inherited var.
STUB_PREAMBLE = (
    '#!/bin/bash\n'
    'O="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
)

# Stub that records everything the delegated ci_local.sh actually receives.
DUMP_STUB = STUB_PREAMBLE + (
    'printf %s "$#" > "$O/_out_argc"\n'
    'if [ "$#" -gt 0 ]; then printf "%s\\0" "$@" > "$O/_out_argv"; else : > "$O/_out_argv"; fi\n'
    'compgen -e | LC_ALL=C sort > "$O/_out_envnames"\n'
    'env > "$O/_out_env"\n'
    'if type pwned >/dev/null 2>&1; then echo YES > "$O/_out_func"; else echo no > "$O/_out_func"; fi\n'
    'printf %s "$PATH" > "$O/_out_path"\n'
    'command -v git > "$O/_out_gitpath" 2>/dev/null || echo NONE > "$O/_out_gitpath"\n'
    'exit 0\n'
)


def _read(path) -> str:
    return Path(path).read_text()


def write_stub(sb: Path, body: str) -> None:
    (sb / "ci" / "ci_local.sh").write_text(body)


def lay_out_repo(path: Path, *, bare: bool, extra_config: str = "") -> None:
    """Create a git repository from plain files (no `git init`, no chmod)."""
    gitdir = path if bare else (path / ".git")
    (gitdir / "objects").mkdir(parents=True, exist_ok=True)
    (gitdir / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    (gitdir / "config").write_text(
        "[core]\n"
        "\trepositoryformatversion = 0\n"
        f"\tbare = {'true' if bare else 'false'}\n"
        + extra_config
    )


def run_wrapper(sb: Path, args, env, timeout: float = 60.0) -> subprocess.CompletedProcess:
    cmd = [*ENTRY, str(sb / "ci" / "ci_local_sanitized.sh"), *args]
    return subprocess.run(
        cmd, cwd=str(sb), env=env, capture_output=True, text=True, timeout=timeout
    )


def parse_env(text: str) -> dict:
    out = {}
    for line in text.split("\n"):
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v
    return out


def read_argv(sb: Path) -> list:
    raw = (sb / "_out_argv").read_bytes()
    if raw == b"":
        return []
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return [p.decode() for p in parts]


def git_read(args, env_overlay=None) -> subprocess.CompletedProcess:
    env = dict(GIT_NEUTRAL)
    if env_overlay:
        env.update(env_overlay)
    return subprocess.run(
        ["/usr/bin/git", *args], env=env, capture_output=True, text=True, timeout=30
    )


class WrapperTests(unittest.TestCase):

    # -- 1 -------------------------------------------------------------------
    def test_01_required_files_nonempty_0644(self):
        """Exactly the two required files exist, are nonempty and 0644 (no chmod)."""
        for p in REQUIRED:
            self.assertTrue(p.is_file(), f"{p} missing")
            self.assertGreater(p.stat().st_size, 0, f"{p} empty")
            self.assertEqual(
                stat.S_IMODE(p.stat().st_mode), 0o644, f"{p} must be mode 0644"
            )
        # Our own delivered files must themselves be whitespace-clean
        # (git diff --check parity), asserted here without invoking chmod.
        for p in REQUIRED:
            for i, line in enumerate(_read(p).split("\n"), 1):
                self.assertEqual(
                    line, line.rstrip(" \t"),
                    f"{p}:{i} has trailing whitespace",
                )

    # -- 2 -------------------------------------------------------------------
    def test_02_bash_syntax_and_set_u_pipefail(self):
        """Wrapper passes `bash -n`, declares set -u/pipefail, and is unbound-safe."""
        src = _read(WRAPPER)
        self.assertIn("set -uo pipefail", src)
        rc = subprocess.run(
            ["/bin/bash", "-n", str(WRAPPER)], capture_output=True, text=True
        )
        self.assertEqual(rc.returncode, 0, rc.stderr)
        # A zero-argument delegation must not trip set -u (no "unbound variable").
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            write_stub(sb, DUMP_STUB)
            cp = run_wrapper(sb, [], env={"PATH": "/usr/bin:/bin"})
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertNotIn("unbound variable", cp.stderr)
            self.assertEqual(_read(sb / "_out_argc"), "0")

    # -- 3 -------------------------------------------------------------------
    def test_03_help_is_local(self):
        """-h/--help is handled locally: exit 0, documents the entrypoint, no delegation."""
        with tempfile.TemporaryDirectory() as td:
            sb = Path(td)
            (sb / "ci").mkdir()
            shutil.copyfile(WRAPPER, sb / "ci" / "ci_local_sanitized.sh")
            # Deliberately NO sibling ci_local.sh: help must not need it.
            for flag in ("--help", "-h"):
                for entry in (["/bin/bash"], ["/bin/bash", "-p"]):
                    cp = subprocess.run(
                        [*entry, str(sb / "ci" / "ci_local_sanitized.sh"), flag],
                        cwd=str(sb), capture_output=True, text=True,
                    )
                    self.assertEqual(cp.returncode, 0, cp.stderr)
                    self.assertIn(
                        "/bin/bash -p ci/ci_local_sanitized.sh", cp.stdout
                    )
                    self.assertNotIn("not found", cp.stderr)

    # -- 4 -------------------------------------------------------------------
    def test_04_entrypoint_documented_no_exec_reliance(self):
        """Supported entrypoint documented; no reliance on the executable bit."""
        src = _read(WRAPPER)
        self.assertIn("/bin/bash -p ci/ci_local_sanitized.sh", src)
        self.assertIn("directly-executed", src)
        # Direct executable invocation is neither claimed nor required: the
        # delivered file carries no execute bits, yet the wrapper works via
        # /bin/bash -p (proven throughout this suite).
        self.assertEqual(WRAPPER.stat().st_mode & 0o111, 0, "must not rely on +x")

    # -- 5 -------------------------------------------------------------------
    def test_05_builtin_path_set_before_any_external_command(self):
        """The first executable statement (a builtin) sets PATH=/usr/bin:/bin."""
        lines = _read(WRAPPER).split("\n")
        first = None
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("#!"):
                continue
            first = s
            break
        self.assertEqual(first, "PATH=/usr/bin:/bin",
                         "PATH must be set by a builtin before anything else")
        idx_path = next(i for i, ln in enumerate(lines) if ln.strip() == "PATH=/usr/bin:/bin")
        idx_env = next((i for i, ln in enumerate(lines) if "env -i" in ln), len(lines))
        self.assertLess(idx_path, idx_env, "PATH set before the env -i transition")

    # -- 6 -------------------------------------------------------------------
    def test_06_hostile_path_cannot_intercept(self):
        """A hostile inherited PATH cannot make the wrapper run planted binaries."""
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            write_stub(sb, DUMP_STUB)
            # Plant hostile look-alikes ahead of the trusted dirs. The wrapper
            # overwrites PATH with a builtin before any external command, so
            # the hostile directory never participates in resolution.
            evil = sb / "evilbin"
            evil.mkdir()
            for name in ("git", "env", "bash", "python3", "grep"):
                (evil / name).write_text('#!/bin/bash\ntouch "$0.RAN"\nexit 0\n')
            env = {"PATH": f"{evil}:/usr/bin:/bin", "HOME": "/tmp"}
            cp = run_wrapper(sb, ["--full"], env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertEqual(_read(sb / "_out_path"), "/usr/bin:/bin",
                             "hostile PATH entry survived into delegation")
            self.assertNotIn(str(evil), _read(sb / "_out_path"))
            self.assertTrue(
                _read(sb / "_out_gitpath").strip().startswith(("/usr/bin/", "/bin/")),
                "git resolved from a trusted directory, not the planted one",
            )
            for name in ("git", "env", "bash", "python3", "grep"):
                self.assertFalse((evil / f"{name}.RAN").exists(),
                                 f"planted {name} was executed")

    # -- 7 -------------------------------------------------------------------
    def test_07_minimal_env_transition_strips_hostile_vars(self):
        """Every inherited variable outside the trusted whitelist is dropped."""
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            write_stub(sb, DUMP_STUB)
            env = {
                "PATH": "/usr/bin:/bin", "HOME": "/tmp",
                "CANARY": "leaked", "FOO": "bar",
                "LD_PRELOAD": "/tmp/evil.so", "LD_LIBRARY_PATH": "/tmp",
                "IFS": ":", "PS4": "+evil ",
            }
            cp = run_wrapper(sb, [], env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            names = set(_read(sb / "_out_envnames").split())
            self.assertTrue(names.issubset(ALLOWED_ENV_NAMES),
                            f"unexpected vars leaked: {names - ALLOWED_ENV_NAMES}")
            for hostile in ("CANARY", "FOO", "LD_PRELOAD", "LD_LIBRARY_PATH", "PS4"):
                self.assertNotIn(hostile, names)
            self.assertNotIn(SEAL_NAME, names, "internal seal must not leak downstream")

    # -- 8 -------------------------------------------------------------------
    def test_08_removes_bash_env_and_exported_functions(self):
        """BASH_ENV/ENV and imported BASH_FUNC_* shell functions are removed."""
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            write_stub(sb, DUMP_STUB)
            env = {
                "PATH": "/usr/bin:/bin", "HOME": "/tmp",
                "BASH_ENV": "/tmp/evil_bashenv",
                "ENV": "/tmp/evil_env",
                "BASH_FUNC_pwned%%": "() { echo PWNED; }",
            }
            cp = run_wrapper(sb, [], env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            names = set(_read(sb / "_out_envnames").split())
            self.assertNotIn("BASH_ENV", names)
            self.assertNotIn("ENV", names)
            self.assertFalse([n for n in names if n.startswith("BASH_FUNC_")])
            self.assertEqual(_read(sb / "_out_func").strip(), "no",
                             "imported function survived the transition")

    # -- 9 -------------------------------------------------------------------
    def test_09_removes_git_routing_and_inline_config(self):
        """Git routing vars and inline GIT_CONFIG_KEY_n/VALUE_n are removed; global/system neutralized."""
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            write_stub(sb, DUMP_STUB)
            env = {
                "PATH": "/usr/bin:/bin", "HOME": "/tmp",
                "GIT_DIR": "/tmp/foreign.git", "GIT_WORK_TREE": "/tmp",
                "GIT_INDEX_FILE": "/tmp/idx", "GIT_OBJECT_DIRECTORY": "/tmp/obj",
                "GIT_COMMON_DIR": "/tmp/common", "GIT_CONFIG": "/tmp/cfg",
                "GIT_CONFIG_COUNT": "3",
                "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": "/tmp/hooks",
                "GIT_CONFIG_KEY_1": "core.fsmonitor", "GIT_CONFIG_VALUE_1": "/tmp/x",
                "GIT_CONFIG_KEY_2": "safe.directory", "GIT_CONFIG_VALUE_2": "*",
            }
            cp = run_wrapper(sb, [], env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            names = set(_read(sb / "_out_envnames").split())
            for hostile in (
                "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                "GIT_COMMON_DIR", "GIT_CONFIG", "GIT_CONFIG_COUNT",
            ):
                self.assertNotIn(hostile, names)
            self.assertFalse([n for n in names if n.startswith("GIT_CONFIG_KEY_")])
            self.assertFalse([n for n in names if n.startswith("GIT_CONFIG_VALUE_")])
            delivered = parse_env(_read(sb / "_out_env"))
            self.assertEqual(delivered.get("GIT_CONFIG_GLOBAL"), "/dev/null")
            self.assertEqual(delivered.get("GIT_CONFIG_SYSTEM"), "/dev/null")
            self.assertEqual(delivered.get("GIT_CONFIG_NOSYSTEM"), "1")

    # -- 10 ------------------------------------------------------------------
    def test_10_global_system_git_config_neutralized(self):
        """A hostile caller HOME/.gitconfig cannot inject Git config into delegation."""
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            hostile_home = sb / "hostilehome"
            hostile_home.mkdir()
            (hostile_home / ".gitconfig").write_text(
                "[core]\n\thooksPath = /tmp/evil_hooks\n"
                "[alias]\n\tevil = !touch /tmp/should_not_run\n"
            )
            write_stub(sb, STUB_PREAMBLE + (
                '{ git config --get core.hooksPath; echo "rc=$?"; } > "$O/_out_hooks" 2>&1\n'
                '{ git config --get alias.evil; echo "rc=$?"; } > "$O/_out_alias" 2>&1\n'
                'env > "$O/_out_env"\n'
                'exit 0\n'
            ))
            env = {"PATH": "/usr/bin:/bin", "HOME": str(hostile_home)}
            cp = run_wrapper(sb, [], env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            hooks = _read(sb / "_out_hooks")
            self.assertNotIn("/tmp/evil_hooks", hooks)
            self.assertIn("rc=1", hooks, "hostile global hooksPath was NOT neutralized")
            self.assertNotIn("should_not_run", _read(sb / "_out_alias"))
            delivered = parse_env(_read(sb / "_out_env"))
            self.assertEqual(delivered.get("HOME"), "/nonexistent")

    # -- 11 ------------------------------------------------------------------
    def test_11_no_caller_home_still_neutralized(self):
        """With no caller HOME and a hostile caller GIT_CONFIG_GLOBAL, config stays neutral."""
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            hostile_cfg = sb / "hostile.gitconfig"
            hostile_cfg.write_text("[core]\n\thooksPath = /tmp/evil_hooks\n")
            write_stub(sb, STUB_PREAMBLE + (
                '{ git config --get core.hooksPath; echo "rc=$?"; } > "$O/_out_hooks" 2>&1\n'
                'env > "$O/_out_env"\n'
                'exit 0\n'
            ))
            # Note: NO HOME in the caller environment; hostile GIT_CONFIG_GLOBAL set.
            env = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_GLOBAL": str(hostile_cfg)}
            cp = run_wrapper(sb, [], env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            hooks = _read(sb / "_out_hooks")
            self.assertNotIn("/tmp/evil_hooks", hooks)
            self.assertIn("rc=1", hooks)
            delivered = parse_env(_read(sb / "_out_env"))
            self.assertEqual(delivered.get("HOME"), "/nonexistent")
            self.assertEqual(delivered.get("GIT_CONFIG_GLOBAL"), "/dev/null")

    # -- 12 ------------------------------------------------------------------
    def test_12_inherited_stage_count_cannot_skip_or_loop(self):
        """Forged stage/count vars cannot skip the transition, keep a function, or loop."""
        for p in REQUIRED:  # fixture asserts 0644 without chmod
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o644)
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            write_stub(sb, DUMP_STUB)
            env = {
                "PATH": "/evil:/usr/bin:/bin", "HOME": "/tmp",
                # forged internal stage/count-like state:
                "__UFCI_SANITIZED_V3": "sealed",
                "__UFCI_SEAL_DEPTH": "999",
                # unbounded-count attack on the inline-config reader:
                "GIT_CONFIG_COUNT": "1000000000",
                "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": "/tmp/h",
                # detectors:
                "CANARY": "leaked",
                "BASH_FUNC_pwned%%": "() { echo PWNED; }",
            }
            # A tight timeout would fire if a forged count drove an unbounded loop.
            cp = run_wrapper(sb, [], env=env, timeout=30)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            names = set(_read(sb / "_out_envnames").split())
            self.assertNotIn("CANARY", names, "forged seal skipped the transition")
            self.assertNotIn(SEAL_NAME, names)
            self.assertFalse([n for n in names if n.startswith("GIT_CONFIG_KEY_")])
            self.assertEqual(_read(sb / "_out_func").strip(), "no",
                             "forged state preserved an imported function")

    # -- 13 ------------------------------------------------------------------
    def test_13_exact_argv_preserved_including_stage_tokens(self):
        """argc/argv are forwarded verbatim, including empty and stage-like tokens."""
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            write_stub(sb, DUMP_STUB)
            args = [
                "--full", "2", "--__ufci_sanitized__", "", "a b\tc",
                "sealed", "__UFCI_SEAL_DEPTH=1", "--gpu", "-x",
            ]
            env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "CANARY": "x"}
            cp = run_wrapper(sb, args, env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertEqual(_read(sb / "_out_argc"), str(len(args)))
            self.assertEqual(read_argv(sb), args)

    # -- 14 ------------------------------------------------------------------
    def test_14_foreign_bare_sentinel_untouched(self):
        """Hostile GIT_DIR is stripped; foreign bare repo is untouched, canonical usable."""
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            lay_out_repo(sb, bare=False)  # canonical non-bare repo at sandbox root
            foreign = sb / "foreign.git"
            lay_out_repo(foreign, bare=True,
                         extra_config="[sentinel]\n\tvalue = original\n")
            (foreign / "SENTINEL").write_bytes(b"SENTINEL-BYTES-DO-NOT-TOUCH")
            cfg_before = (foreign / "config").read_bytes()
            sen_before = hashlib.sha256((foreign / "SENTINEL").read_bytes()).hexdigest()

            write_stub(sb, STUB_PREAMBLE + (
                'git rev-parse --git-dir > "$O/_gitdir" 2>"$O/_giterr"\n'
                'git rev-parse --is-bare-repository > "$O/_isbare" 2>>"$O/_giterr"\n'
                'exit 0\n'
            ))
            env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp",
                   "GIT_DIR": str(foreign), "GIT_WORK_TREE": "/tmp"}
            cp = run_wrapper(sb, [], env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr)

            # Delegated git resolved the canonical repo, not the foreign bare.
            self.assertEqual(_read(sb / "_isbare").strip(), "false")
            self.assertNotIn("foreign.git", _read(sb / "_gitdir"))

            # Foreign bare bytes/config unchanged.
            self.assertEqual((foreign / "config").read_bytes(), cfg_before)
            self.assertEqual(
                hashlib.sha256((foreign / "SENTINEL").read_bytes()).hexdigest(),
                sen_before,
            )
            self.assertEqual(
                git_read(["config", "--file", str(foreign / "config"),
                          "--get", "sentinel.value"]).stdout.strip(),
                "original",
            )
            # Canonical repo still usable / non-bare.
            self.assertEqual(
                git_read(["--git-dir", str(sb / ".git"),
                          "rev-parse", "--is-bare-repository"]).stdout.strip(),
                "false",
            )

    # -- 15 ------------------------------------------------------------------
    def test_15_foreign_bare_untouched_under_safe_bare_explicit_policy(self):
        """Same guarantees hold, host-policy-independently, under safe.bareRepository=explicit."""
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            lay_out_repo(sb, bare=False)
            foreign = sb / "foreign.git"
            lay_out_repo(foreign, bare=True,
                         extra_config="[sentinel]\n\tvalue = original\n")
            (foreign / "SENTINEL").write_bytes(b"SENTINEL-BYTES-DO-NOT-TOUCH")
            cfg_before = (foreign / "config").read_bytes()

            # Deterministically activate the policy (independent of the host).
            policy = sb / "policy.gitconfig"
            policy.write_text("[safe]\n\tbareRepository = explicit\n")
            pol = {"GIT_CONFIG_GLOBAL": str(policy)}

            # The inspection helper stays usable under the policy: explicit
            # --git-dir works, while auto-discovery is refused.
            explicit = git_read(
                ["--git-dir", str(foreign), "rev-parse", "--is-bare-repository"],
                env_overlay=pol,
            )
            self.assertEqual(explicit.returncode, 0, explicit.stderr)
            self.assertEqual(explicit.stdout.strip(), "true")
            auto = git_read(["-C", str(foreign), "rev-parse", "--is-bare-repository"],
                            env_overlay=pol)
            self.assertNotEqual(auto.returncode, 0)
            self.assertIn("safe.bareRepository", auto.stderr)

            # Now drive the wrapper with the policy active AND a hostile GIT_DIR.
            write_stub(sb, STUB_PREAMBLE + (
                'git rev-parse --is-bare-repository > "$O/_isbare" 2>"$O/_giterr"\n'
                'exit 0\n'
            ))
            env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp",
                   "GIT_CONFIG_GLOBAL": str(policy), "GIT_DIR": str(foreign)}
            cp = run_wrapper(sb, [], env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertEqual(_read(sb / "_isbare").strip(), "false")

            # Foreign bare unchanged; canonical usable via explicit, policy-safe read.
            self.assertEqual((foreign / "config").read_bytes(), cfg_before)
            self.assertEqual(
                git_read(["--git-dir", str(foreign), "config",
                          "--get", "sentinel.value"], env_overlay=pol).stdout.strip(),
                "original",
            )
            self.assertEqual(
                git_read(["--git-dir", str(sb / ".git"),
                          "rev-parse", "--is-bare-repository"],
                         env_overlay=pol).stdout.strip(),
                "false",
            )

    # -- 16 ------------------------------------------------------------------
    def test_16_untracked_trailing_whitespace_delegated_checker(self):
        """A delegated whitespace checker flags a real untracked trailing-space+tab file."""
        checker = STUB_PREAMBLE + (
            'cd "$O"\n'
            'hits="$(grep -rnP "[ \\t]+\\$" . 2>/dev/null || true)"\n'
            'if [ -n "$hits" ]; then\n'
            '  printf "%s\\n" "$hits"\n'
            '  echo "TRAILING-WHITESPACE-FOUND"\n'
            '  exit 1\n'
            'fi\n'
            'exit 0\n'
        )
        with tempfile.TemporaryDirectory() as td:
            sb = make_sandbox_from(td)
            write_stub(sb, checker)
            offender = sb / "offending_file.txt"
            offender.write_text(
                "trailing spaces here   \nand tabs here\t\t\nclean line\n"
            )
            env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
            cp = run_wrapper(sb, [], env=env)
            self.assertNotEqual(cp.returncode, 0, "checker should fail on trailing ws")
            self.assertIn("offending_file.txt", cp.stdout)
            self.assertIn("TRAILING-WHITESPACE-FOUND", cp.stdout)

            # Clean it up and prove the tree fixture is now clean.
            offender.unlink()
            cp2 = run_wrapper(sb, [], env=env)
            self.assertEqual(cp2.returncode, 0, cp2.stdout + cp2.stderr)


def make_sandbox_from(td: str) -> Path:
    sb = Path(td)
    (sb / "ci").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WRAPPER, sb / "ci" / "ci_local_sanitized.sh")
    return sb


if __name__ == "__main__":
    unittest.main(verbosity=2)
