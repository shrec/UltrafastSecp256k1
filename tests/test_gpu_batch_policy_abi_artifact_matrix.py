"""test_gpu_batch_policy_abi_artifact_matrix.py -- task
GPU_ABI_MATRIX_PROGRAM_CLAUDE_561.

pytest wrapper around audit/gpu_batch_policy_abi_artifact_matrix.py.  The
dependency-hash test always runs (it needs no toolchain).  The full-matrix
tests build, link, load and run real native artifacts, so they are skipped
-- not faked as passing -- when cc/c++/ar are not on PATH.
"""

import fcntl
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MATRIX_PATH = os.path.join(REPO, "audit", "gpu_batch_policy_abi_artifact_matrix.py")


def _load_matrix():
    spec = importlib.util.spec_from_file_location(
        "gpu_batch_policy_abi_artifact_matrix", MATRIX_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


matrix = _load_matrix()

TOOLCHAIN_AVAILABLE = all(
    shutil.which(exe) is not None
    for exe in (matrix.CC, matrix.CXX, matrix.AR))
requires_toolchain = pytest.mark.skipif(
    not TOOLCHAIN_AVAILABLE,
    reason="cc/c++/ar not all available on PATH: "
           f"{ {'CC': matrix.CC, 'CXX': matrix.CXX, 'AR': matrix.AR} }")

# --scratch-root now must be strictly beneath the authorized repo out/ tree
# (matrix._validate_scratch_root) -- pytest's own tmp_path fixture (system
# temp, e.g. /tmp) is external and would be rejected, so the CLI-driven
# full-matrix tests use this repo-local scratch space instead.
PYTEST_SCRATCH_BASE = os.path.join(
    matrix.AUTHORIZED_SCRATCH_PARENT, ".tasks", matrix.TASK_ID, "pytest_scratch")


@pytest.fixture
def repo_scratch_root():
    """A fresh, repo-local scratch directory strictly beneath the authorized
    out/ tree -- what --scratch-root now requires -- cleaned up afterward."""
    os.makedirs(PYTEST_SCRATCH_BASE, exist_ok=True)
    path = tempfile.mkdtemp(dir=PYTEST_SCRATCH_BASE)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _cleanup_sealed_deps():
    """Review rework (HIGH -- TOCTOU): pin_verified_dependencies() now holds
    open, sealed memfds in the module-global matrix._SEALED_DEPS for the
    lifetime of whatever pinned them. main() closes and clears these itself
    in its own finally block, but tests that call pin_verified_dependencies()
    directly (bypassing main(), as most unit tests in this file do) must not
    leak fds or bleed sealed state into later tests. This autouse fixture
    guarantees both for every test in this file, so individual tests don't
    each need to repeat the same try/finally boilerplate."""
    yield
    for entry in list(matrix._SEALED_DEPS.values()):
        try:
            os.close(entry["fd"])
        except OSError:
            pass
    matrix._SEALED_DEPS.clear()


def test_dependency_hashes_match_accepted():
    """Fail-closed: every accepted-artifact dependency this task reuses must
    still hash to the value gpu_batch_policy_abi_core_check.py already
    pinned -- catches silent drift without needing a compiler."""
    for path, want in matrix.ACCEPTED_SHA.items():
        assert os.path.isfile(path), f"missing dependency: {path}"
        assert not os.path.islink(path), f"dependency must not be a symlink: {path}"
        got = matrix.sha256_file(path)
        assert got == want, f"dependency byte drift: {path} (want {want} got {got})"


# --------------------------------------------------------------------------- #
# Review rework (MEDIUM): verify_dependencies() previously hashed by         #
# pathname, closed the file, and returned only the hash -- every later       #
# consumer (load_core_check(), compilers) reopened the same pathname a       #
# second time, leaving a real TOCTOU window for a symlink/rename             #
# substitution race. verify_dependencies() now reads with O_NOFOLLOW and     #
# returns bytes.                                                             #
# --------------------------------------------------------------------------- #
def test_verify_dependencies_rejects_symlinked_input(tmp_path, monkeypatch):
    """A symlink swapped in for one of the accepted dependency paths must be
    rejected atomically via O_NOFOLLOW at open time -- not via a separate
    os.path.islink() check that a rename race could still slip past between
    the check and a second, independent open-by-name."""
    decoy = tmp_path / "decoy_oracle.c"
    decoy.write_text("// decoy, not the accepted oracle\n")
    symlinked_path = tmp_path / "symlinked_dependency"
    os.symlink(str(decoy), str(symlinked_path))
    monkeypatch.setattr(matrix, "ACCEPTED_SHA", {str(symlinked_path): "0" * 64})
    with pytest.raises(SystemExit):
        matrix.verify_dependencies()


# --------------------------------------------------------------------------- #
# Review rework (HIGH -- TOCTOU): a bookend hash check around a reopened     #
# *pathname* (the pre-rework pin_verified_dependencies()/reverify_pinned_    #
# dependency()/compile_cmd()/load_core_check() design) cannot see a same-UID #
# swap that happens strictly inside the compiler/importlib consumption       #
# window and is undone before the "after" hash runs. Every dependency is now #
# delivered from a memfd sealed F_SEAL_WRITE|SHRINK|GROW|SEAL -- the kernel  #
# refuses any further write to that inode, from any process, ever. "argv"-   #
# kind dependencies (everything except the header) are referenced only by    #
# "/dev/fd/<fd>": there is no named pathname for them to race at all. The    #
# header alone needs a real, -I-searchable name (a symlink to               #
# "/proc/self/fd/<fd>", not a hardlink -- memfd inodes cannot be hardlinked  #
# across the anonymous-memfd/real-filesystem device boundary, EXDEV); its    #
# use-time re-verification checks resolved inode identity, not just content  #
# hash, so a same-UID attacker's only available "restore" (a fresh regular   #
# file carrying the public, byte-identical accepted bytes -- they cannot     #
# recreate a symlink to this process's own sealed fd) is still detected.     #
# --------------------------------------------------------------------------- #
def test_pin_verified_dependencies_delivers_sealed_kernel_immutable_fds(tmp_path):
    """Every accepted dependency -- including the header, which round 1 of
    this fix still gave a named symlink entry (see _SEALED_DEPS's module
    docstring for why that was still substitutable) -- is delivered
    uniformly from a fd whose kernel write seal (not a bookend hash check)
    is what makes it immutable: a direct write attempt to the sealed fd
    must fail with EPERM, its content read back via _reverify_sealed_dep()
    (never a reopened pathname) must match what verify_dependencies()
    verified, its reference is always "/dev/fd/<fd>" with no named,
    directory-visible path anywhere, and the real, accepted repo files
    (victim/path safety) must remain untouched by any of this."""
    build_dir = str(tmp_path / "pin_sealed")
    os.makedirs(build_dir, exist_ok=True)
    verified = matrix.verify_dependencies()
    pinned = matrix.pin_verified_dependencies(build_dir, verified)
    assert set(pinned) == set(matrix.ACCEPTED_SHA)
    for original_path, ref in pinned.items():
        entry = matrix._SEALED_DEPS[ref]
        assert entry["sha256"] == matrix.ACCEPTED_SHA[original_path]
        got_seals = fcntl.fcntl(entry["fd"], matrix.F_GET_SEALS)
        assert got_seals & matrix._FULL_WRITE_SEAL == matrix._FULL_WRITE_SEAL
        assert matrix._reverify_sealed_dep(ref) == verified[original_path]
        with pytest.raises(OSError):
            os.write(entry["fd"], b"attacker-controlled bytes")
        assert ref == f"/dev/fd/{entry['fd']}"
    for original_path, want in matrix.ACCEPTED_SHA.items():
        assert matrix.sha256_file(original_path) == want


def test_sealed_memfd_fds_use_mfd_cloexec_at_creation():
    """No descriptor leakage: every sealed memfd is created with
    MFD_CLOEXEC, so it is never accidentally inherited by an unrelated
    subprocess this module spawns (only compile_cmd()'s explicit pass_fds
    can hand a specific one to a specific child) -- verified here directly
    against the raw fd flags, independent of any subprocess behavior."""
    fd = matrix._create_sealed_memfd("probe", b"probe bytes\n")
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        assert flags & fcntl.FD_CLOEXEC
    finally:
        os.close(fd)


def test_reverify_sealed_dep_write_attempt_fails_closed_structurally(tmp_path):
    """Substitution-race mutant for a COMPILED input (ORACLE_C):
    pin_verified_dependencies() delivers it from a sealed fd, not an
    owner-writable regular file, so there is no in-place overwrite for an
    attacker to perform at all -- attempting one via the fd directly (the
    strongest capability an in-process attacker could have; a real,
    separate same-UID process cannot even obtain this fd -- see
    test_sealed_memfd_fds_use_mfd_cloexec_at_creation and
    test_compile_cmd_passes_only_referenced_sealed_fds_to_subprocess) must
    fail closed with OSError, and _reverify_sealed_dep() must still return
    the untouched, correct bytes afterward."""
    build_dir = str(tmp_path / "compiled_input_substitution")
    os.makedirs(build_dir, exist_ok=True)
    verified = matrix.verify_dependencies()
    pinned = matrix.pin_verified_dependencies(build_dir, verified)
    oracle_ref = pinned[matrix.ORACLE_C]
    fd = matrix._SEALED_DEPS[oracle_ref]["fd"]
    with pytest.raises(OSError):
        os.write(fd, b"/* substituted by attacker between pin and compile */")
    assert matrix._reverify_sealed_dep(oracle_ref) == verified[matrix.ORACLE_C]


def test_load_core_check_sealed_fd_write_attempt_fails_closed(tmp_path):
    """Substitution-race mutant for the PYTHON input (CORE_CHECK_PATH):
    pin_verified_dependencies() delivers it from a sealed fd; a direct write
    attempt against it must fail with OSError, and load_core_check() --
    which now execs bytes read straight from that sealed fd, never a
    reopened pathname -- must still return the correct, unmodified module."""
    build_dir = str(tmp_path / "python_input_substitution")
    os.makedirs(build_dir, exist_ok=True)
    verified = matrix.verify_dependencies()
    pinned = matrix.pin_verified_dependencies(build_dir, verified)
    core_check_ref = pinned[matrix.CORE_CHECK_PATH]
    fd = matrix._SEALED_DEPS[core_check_ref]["fd"]
    with pytest.raises(OSError):
        os.write(fd, b"\n# substituted after pin, before use\n")
    matrix._RESOLVED_DEPS["CORE_CHECK_PATH"] = core_check_ref
    try:
        mod = matrix.load_core_check()
        assert hasattr(mod, "generate_cases")
        assert hasattr(mod, "ORACLE_MUTANTS")
    finally:
        matrix._RESOLVED_DEPS.pop("CORE_CHECK_PATH", None)


@requires_toolchain
def test_compile_cmd_passes_only_referenced_sealed_fds_to_subprocess(tmp_path):
    """No descriptor leakage, scoped: compile_cmd() must pass_fds only the
    fd numbers of the sealed "/dev/fd/<fd>" references that literally
    appear in that specific invocation's argv. An `ar` invocation over an
    already-compiled .o file (no "/dev/fd/<fd>" token at all -- the header
    is never a special case any more, see _SEALED_DEPS's module docstring)
    must receive an empty pass_fds, even though several sealed dependencies
    remain open in this process for the lifetime of the pin."""
    build_dir = str(tmp_path / "fd_scope_probe")
    os.makedirs(build_dir, exist_ok=True)
    verified = matrix.verify_dependencies()
    matrix.pin_verified_dependencies(build_dir, verified)
    core_check = matrix.load_core_check()
    _static_r, _lib_a, obj = matrix.build_static_matrix(build_dir, core_check)

    captured = {}
    real_run = matrix.subprocess.run

    def spy_run(argv, **kwargs):
        captured["pass_fds"] = kwargs.get("pass_fds")
        return real_run(argv, **kwargs)

    matrix.subprocess.run = spy_run
    try:
        lib_a2 = os.path.join(build_dir, "liboracle_fd_scope_probe.a")
        if os.path.exists(lib_a2):
            os.remove(lib_a2)
        matrix.compile_cmd([matrix.AR, "rcsD", lib_a2, obj])
    finally:
        matrix.subprocess.run = real_run

    assert captured["pass_fds"] == []


@requires_toolchain
def test_compile_cmd_pass_fds_includes_exactly_the_dev_fd_tokens_in_argv(tmp_path):
    """Positive companion to the test above: when an argv genuinely contains
    a sealed "/dev/fd/<fd>" token (a header-inlined source from
    _seal_inlined_source()), compile_cmd() must include exactly that fd
    number in pass_fds -- not every currently-open sealed dependency, and
    not none."""
    build_dir = str(tmp_path / "fd_scope_probe_positive")
    os.makedirs(build_dir, exist_ok=True)
    ref = matrix._seal_inlined_source(
        "probe.c",
        b'#include "gpu_batch_policy_abi_contract.h"\nint main(void){return 0;}\n')
    fd = matrix._SEALED_DEPS[ref]["fd"]

    captured = {}
    real_run = matrix.subprocess.run

    def spy_run(argv, **kwargs):
        captured["pass_fds"] = kwargs.get("pass_fds")
        return real_run(argv, **kwargs)

    matrix.subprocess.run = spy_run
    try:
        obj = os.path.join(build_dir, "probe.o")
        matrix.compile_cmd([matrix.CC, "-std=c11", "-O0", "-fPIC", "-c",
                            "-x", "c", ref, "-o", obj])
    finally:
        matrix.subprocess.run = real_run

    assert captured["pass_fds"] == [fd]


@requires_toolchain
def test_header_swap_and_exact_original_symlink_restore_has_no_effect_on_real_compile(
        tmp_path):
    """Flagship regression for the review's strongest attack against round 1
    of this fix (a named symlink to "/proc/self/fd/<fd>" for the header,
    re-verified by resolved-inode identity before/after each compile): a
    same-UID attacker swaps that named entry for a plain regular file
    carrying malicious content, lets a real compiler consume it, then
    recreates the *exact original symlink text* -- not a byte-identical
    regular file (a different, unsealed inode round 1's check already
    caught), but the literal "/proc/self/fd/<fd>" string itself. That text
    is not process-specific secret data: when the *verifying* process (not
    the compiler that already ran) resolves "self" a second time, it always
    means itself, and its own sealed fd is still open at that same number
    the whole time -- so restoring the literal original symlink always
    "passes" a resolved-inode check performed by the process that owns the
    fd, regardless of what a *different* process actually opened through
    the name in between. See _SEALED_DEPS's module docstring.

    The fix (_seal_inlined_source()/_compile_ready_source()) gives the
    header no named, directory-visible entry at all, ever. This test proves
    that directly: it builds the exact legacy attack surface by hand (a
    directory entry holding a symlink to "/proc/self/fd/<header_fd>",
    precisely what round 1 used to create and rely on), confirms a real
    "legacy-style" -I-search compile genuinely consumes the swapped-in
    malicious content during the window (a real, load-bearing exploit, not
    a strawman), then runs the *fixed* construction through the identical
    swap-then-restore-original-symlink window and shows its output reflects
    only the original, verified header bytes -- the swap and the "restored"
    original symlink text both have zero effect on it, because nothing in
    the fixed construction ever resolves that name. Victim safety (the real
    accepted header on disk is never touched) and no fd leaks (every sealed
    fd this test opens, including the header's and the fixed probe's own,
    is closed by the _cleanup_sealed_deps autouse fixture above) both hold
    throughout."""
    build_dir = str(tmp_path / "header_swap_exact_symlink_restore")
    os.makedirs(build_dir, exist_ok=True)
    verified = matrix.verify_dependencies()
    pinned = matrix.pin_verified_dependencies(build_dir, verified)
    header_fd = matrix._SEALED_DEPS[pinned[matrix.HEADER]]["fd"]
    accepted_header_bytes = verified[matrix.HEADER]
    matrix._RESOLVED_DEPS["HEADER_REF"] = pinned[matrix.HEADER]

    # Recreate the exact legacy attack surface by hand: a directory entry
    # named after the header, holding a symlink whose text is the same
    # "/proc/self/fd/<fd>" string round 1 of this fix used to create and
    # rely on. Nothing in the fixed pipeline below ever opens this path --
    # building it precisely is what makes this a real regression test for
    # the exact bypass the review described, not a strawman.
    legacy_deps_dir = os.path.join(build_dir, "_legacy_named_header_probe")
    os.makedirs(legacy_deps_dir, exist_ok=True)
    legacy_header_path = os.path.join(
        legacy_deps_dir, "gpu_batch_policy_abi_contract.h")
    original_symlink_target = f"/proc/self/fd/{header_fd}"
    os.symlink(original_symlink_target, legacy_header_path)

    malicious_header_bytes = accepted_header_bytes.replace(
        b"#define", b"#define ATTACKER_INJECTED 1\n#define", 1)
    assert malicious_header_bytes != accepted_header_bytes

    probe_text = (
        '#include "gpu_batch_policy_abi_contract.h"\n'
        '#ifdef ATTACKER_INJECTED\n#error consumed malicious header\n#endif\n'
        'int x;\n')
    real_run = matrix.subprocess.run

    def swap_malicious_then_restore_exact_original_symlink(argv, **kwargs):
        os.remove(legacy_header_path)
        with open(legacy_header_path, "wb") as f:
            f.write(malicious_header_bytes)
        try:
            return real_run(argv, **kwargs)
        finally:
            # Restore the *exact original* symlink text -- the strongest
            # form of the attack: not a byte-identical regular file (a
            # different, unsealed inode -- already caught by round 1's
            # inode-identity check), but the literal original
            # "/proc/self/fd/<fd>" string, which trivially resolves back to
            # this process's own still-open sealed fd regardless of what a
            # different process actually read through it in between.
            os.remove(legacy_header_path)
            os.symlink(original_symlink_target, legacy_header_path)

    matrix.subprocess.run = swap_malicious_then_restore_exact_original_symlink
    try:
        legacy_src = matrix.write_src(
            build_dir, "swap_probe_legacy.c", probe_text)
        legacy_obj = os.path.join(build_dir, "swap_probe_legacy.o")
        legacy_ok, legacy_out = matrix.compile_cmd(
            [matrix.CC, "-std=c11", "-O0", "-fPIC", "-c",
             "-I", legacy_deps_dir, legacy_src, "-o", legacy_obj],
            must_succeed=False)

        fixed_ref = matrix._seal_inlined_source(
            "swap_probe_fixed.c", probe_text.encode())
        fixed_obj = os.path.join(build_dir, "swap_probe_fixed.o")
        fixed_ok, fixed_out = matrix.compile_cmd(
            [matrix.CC, "-std=c11", "-O0", "-fPIC", "-c", "-x", "c",
             fixed_ref, "-o", fixed_obj],
            must_succeed=False)
    finally:
        matrix.subprocess.run = real_run
        matrix._RESOLVED_DEPS.pop("HEADER_REF", None)
        if os.path.islink(legacy_header_path) or os.path.exists(legacy_header_path):
            os.remove(legacy_header_path)
        shutil.rmtree(legacy_deps_dir, ignore_errors=True)

    # (a) precondition: the legacy -I-search delivery really did consume
    # the swapped-in malicious content during the window -- a real,
    # load-bearing attack, not a strawman.
    assert not legacy_ok
    assert "consumed malicious header" in legacy_out, (
        "precondition invalid: the legacy -I path did not actually "
        "consume the swapped-in malicious header during the window")

    # (b) the fix: run through the identical swap/restore-exact-symlink
    # window, the header-inlined sealed construction never resolves any
    # named path for the header, so it compiles clean against only the
    # original, verified bytes -- the malicious swap and the "restored"
    # original symlink text both had zero effect on it.
    assert fixed_ok, f"fixed compile unexpectedly failed: {fixed_out}"
    assert "consumed malicious header" not in fixed_out
    assert os.path.isfile(fixed_obj)

    # victim safety: the real, on-disk accepted header was never touched.
    assert matrix.sha256_file(matrix.HEADER) == matrix.ACCEPTED_SHA[matrix.HEADER]


@requires_toolchain
def test_full_matrix_uses_pinned_dependency_copies_not_original_paths(tmp_path):
    """End-to-end: a real main() run must resolve ORACLE_C/NEGATIVE_C/
    CORE_CHECK_PATH/DEP_CONSUMER_C/DEP_CONSUMER_CPP/HEADER_REF to sealed,
    pinned references (via _RESOLVED_DEPS) for the duration of the call, and
    _RESOLVED_DEPS/_SEALED_DEPS must be cleared (and every sealed fd closed)
    again once main() returns -- so a later, unrelated direct build_*() call
    in the same process (as other tests in this file perform) falls back to
    the original module-level constants rather than a stale, no-longer-valid
    fd/scratch path from a run that already finished. Review rework (HIGH --
    TOCTOU): ORACLE_C/CORE_CHECK_PATH now resolve to "/dev/fd/<fd>" sealed-fd
    references, not a pathname beneath a pinned scratch build_dir."""
    run_dir = os.path.join(PYTEST_SCRATCH_BASE, "resolved_deps_probe")
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(PYTEST_SCRATCH_BASE, exist_ok=True)

    captured = {}
    real_load_core_check = matrix.load_core_check

    def spy_load_core_check():
        captured["oracle_c"] = matrix._dep("ORACLE_C", matrix.ORACLE_C)
        captured["core_check_path"] = matrix._dep(
            "CORE_CHECK_PATH", matrix.CORE_CHECK_PATH)
        return real_load_core_check()

    orig = matrix.load_core_check
    matrix.load_core_check = spy_load_core_check
    try:
        argv_orig = sys.argv
        sys.argv = ["gpu_batch_policy_abi_artifact_matrix.py",
                   "--scratch-root", run_dir]
        try:
            rc = matrix.main()
        finally:
            sys.argv = argv_orig
    finally:
        matrix.load_core_check = orig
        shutil.rmtree(run_dir, ignore_errors=True)

    assert rc == 0
    assert captured["oracle_c"] != matrix.ORACLE_C
    assert captured["oracle_c"].startswith("/dev/fd/")
    assert captured["core_check_path"] != matrix.CORE_CHECK_PATH
    assert captured["core_check_path"].startswith("/dev/fd/")
    # cleared after main() returns -- no leakage into later direct calls,
    # and no descriptor leakage (every sealed fd main() opened is closed).
    assert matrix._RESOLVED_DEPS == {}
    assert matrix._SEALED_DEPS == {}


# --------------------------------------------------------------------------- #
# Review rework (HIGH -- TOCTOU, round 2): INCLUDE_DIR / the pinned header    #
# symlink directory no longer exist at all -- every compiled source gets the #
# header inlined directly into its own sealed blob instead (see              #
# _seal_inlined_source()/_compile_ready_source() and _SEALED_DEPS's module   #
# docstring). _shared_object_compile_prefix() carries no -I flag any more.   #
# --------------------------------------------------------------------------- #
def test_shared_object_compile_prefix_has_no_include_search_flag():
    prefix = matrix._shared_object_compile_prefix()
    assert prefix == [matrix.CC, "-std=c11", "-O0", "-fPIC", "-shared"]
    assert "-I" not in prefix


def test_shared_object_compile_prefix_inserts_extra_flags_before_shared():
    prefix = matrix._shared_object_compile_prefix(
        extra_flags=["-fvisibility=hidden"])
    assert prefix == [matrix.CC, "-std=c11", "-O0", "-fPIC",
                      "-fvisibility=hidden", "-shared"]


# --------------------------------------------------------------------------- #
# _inline_header_bytes() / _get_verified_header_bytes() / _seal_inlined_     #
# source(): the header-inlining mechanism that replaced the named symlink.   #
# --------------------------------------------------------------------------- #
def test_inline_header_bytes_replaces_include_line_with_header_content():
    source = b'#include "gpu_batch_policy_abi_contract.h"\nint x;\n'
    header = b"#define LB_PROBE_MARKER 1\n"
    combined = matrix._inline_header_bytes(source, header, "probe.c")
    assert matrix._HEADER_INCLUDE_LINE not in combined
    assert header in combined
    assert b"int x;" in combined


def test_inline_header_bytes_is_a_noop_passthrough_when_include_line_absent():
    """Deliberately-broken probe sources (e.g. a forced-compile-failure test
    fixture with no header include at all) must not be turned into an
    unrelated FAIL-CLOSED abort just for lacking the marker -- see
    _inline_header_bytes()'s docstring. The compile still fails for its own
    reasons (here, invalid syntax), never a silent false pass."""
    source = b"this is not valid freestanding C syntax {{{ ???\n"
    header = b"#define LB_PROBE_MARKER 1\n"
    assert matrix._inline_header_bytes(source, header, "broken.c") == source


def test_inline_header_bytes_is_a_noop_when_include_line_appears_more_than_once():
    source = (b'#include "gpu_batch_policy_abi_contract.h"\n'
             b'#include "gpu_batch_policy_abi_contract.h"\n')
    header = b"#define LB_PROBE_MARKER 1\n"
    assert matrix._inline_header_bytes(source, header, "dup.c") == source


def test_get_verified_header_bytes_falls_back_to_real_path_when_nothing_pinned():
    assert matrix._RESOLVED_DEPS.get("HEADER_REF") is None
    assert (matrix._get_verified_header_bytes()
            == matrix._read_nofollow(matrix.HEADER))


def test_seal_inlined_source_produces_dev_fd_reference_with_inlined_content():
    ref = matrix._seal_inlined_source(
        "probe.c", b'#include "gpu_batch_policy_abi_contract.h"\nint x;\n')
    assert ref.startswith("/dev/fd/")
    data = matrix._reverify_sealed_dep(ref)
    assert matrix._HEADER_INCLUDE_LINE not in data
    assert b"int x;" in data
    assert matrix._read_nofollow(matrix.HEADER) in data


@requires_toolchain
def test_full_matrix_shared_dlopen_consumer_compiles_use_sealed_source_not_core_root(
        repo_scratch_root):
    """Review rework: build_shared_matrix()'s dlopen C/C++ consumer compiles
    previously hardcoded the bare CORE_ROOT constant as their -I directory
    while every other accepted-header consumer in this module went through
    the pinned header -- CORE_ROOT (out/.tasks/GPU_ABI_NATIVE_CORE_264/
    prototype) is never hash-pinned, so a header substituted there could
    have reached exactly these two compiles without tripping any dependency
    check. They now go through _compile_ready_source() like every other
    accepted source in this module: the recorded dlopen-consumer compile
    commands must reference a sealed "/dev/fd/<fd>" token, never the
    original CORE_ROOT path, and carry no -I flag at all."""
    run_dir = os.path.join(repo_scratch_root, "sealed_dlopen_consumer_source")
    argv_orig = sys.argv
    sys.argv = ["gpu_batch_policy_abi_artifact_matrix.py",
               "--scratch-root", run_dir]
    try:
        rc = matrix.main()
    finally:
        sys.argv = argv_orig
    assert rc == 0

    dlopen_consumer_cmds = [
        c for c in matrix._commands if "dlopen_consumer_c" in c]
    assert len(dlopen_consumer_cmds) == 2  # the C and the C++ consumer
    for cmd in dlopen_consumer_cmds:
        assert matrix.CORE_ROOT not in cmd
        assert " -I " not in f" {cmd} "
        assert any(tok.startswith("/dev/fd/") for tok in cmd.split())


def test_elf_parser_rejects_non_elf(tmp_path):
    bogus = tmp_path / "not_elf.bin"
    bogus.write_bytes(b"not an elf file at all, just plain bytes")
    with pytest.raises(ValueError):
        matrix.Elf(str(bogus))


def test_archive_parser_rejects_bad_magic(tmp_path):
    bogus = tmp_path / "not_an_archive.a"
    bogus.write_bytes(b"garbage, not !<arch>\\n")
    with pytest.raises(ValueError):
        matrix.parse_archive(str(bogus))


# --------------------------------------------------------------------------- #
# _evaluate_needed_ok: pure decision-function coverage of the exact-one-      #
# oracle / no-duplicate / no-surplus-system NEEDED contract. Needs no         #
# toolchain -- these exercise the decision logic directly against            #
# constructed NEEDED lists, including a genuine duplicate-entry case that a   #
# real linker will not reliably reproduce (GNU ld deduplicates identical      #
# `-l:` sonames on the command line), so the logic is still verified.         #
# --------------------------------------------------------------------------- #
def test_evaluate_needed_ok_accepts_oracle_plus_allowed_libc():
    r = matrix._evaluate_needed_ok(["libc.so.6", "liboracle.so.1"], "liboracle.so.1")
    assert r["ok"]
    assert r["oracle_count"] == 1
    assert r["duplicates"] == []
    assert r["unexpected_system_needed"] == []


def test_evaluate_needed_ok_accepts_oracle_only_no_system_dep():
    r = matrix._evaluate_needed_ok(["liboracle.so.1"], "liboracle.so.1")
    assert r["ok"]
    assert r["oracle_count"] == 1


def test_evaluate_needed_ok_rejects_duplicate_oracle_needed():
    r = matrix._evaluate_needed_ok(
        ["liboracle.so.1", "liboracle.so.1", "libc.so.6"], "liboracle.so.1")
    assert not r["ok"]
    assert r["duplicates"] == ["liboracle.so.1"]


def test_evaluate_needed_ok_rejects_duplicate_system_needed():
    r = matrix._evaluate_needed_ok(
        ["liboracle.so.1", "libc.so.6", "libc.so.6"], "liboracle.so.1")
    assert not r["ok"]
    assert r["duplicates"] == ["libc.so.6"]


def test_evaluate_needed_ok_rejects_surplus_system_dependency():
    r = matrix._evaluate_needed_ok(
        ["liboracle.so.1", "libc.so.6", "libm.so.6"], "liboracle.so.1")
    assert not r["ok"]
    assert r["unexpected_system_needed"] == ["libm.so.6"]


def test_evaluate_needed_ok_rejects_missing_oracle():
    r = matrix._evaluate_needed_ok(["libc.so.6"], "liboracle.so.1")
    assert not r["ok"]
    assert r["oracle_count"] == 0


def test_evaluate_needed_ok_rejects_wrong_oracle_soname():
    r = matrix._evaluate_needed_ok(
        ["libc.so.6", "liboracle_decoy.so.1"], "liboracle.so.1")
    assert not r["ok"]
    assert r["oracle_count"] == 0


@requires_toolchain
def test_full_artifact_matrix_run1(repo_scratch_root):
    """First of the two required complete matrix invocations."""
    # main() parses sys.argv, so drive it the same way the CLI does.
    run_dir = os.path.join(repo_scratch_root, "run1")
    argv_orig = sys.argv
    sys.argv = ["gpu_batch_policy_abi_artifact_matrix.py",
               "--scratch-root", run_dir]
    try:
        rc = matrix.main()
    finally:
        sys.argv = argv_orig
    assert rc == 0
    report = os.path.join(run_dir, "report.md")
    assert os.path.isfile(report)
    with open(report) as f:
        text = f.read()
    assert "**overall** | **PASS**" in text
    # task551_mutants_ok must be part of the load-bearing gate table the full
    # CLI matrix prints -- not only a direct-pytest-only assertion -- so a
    # regression here fails both invocation surfaces the manager required.
    assert "| task551_mutants_ok | PASS |" in text
    assert "| task551_ok | PASS |" in text
    assert "| static_archive_mutants_ok | PASS |" in text
    assert "| i386_ok | PASS |" in text


@requires_toolchain
def test_full_artifact_matrix_run2_independent(repo_scratch_root):
    """Second, independent complete matrix invocation (fresh scratch dir) --
    proves the matrix is not order-dependent on a prior run's process state
    (e.g. no accidental RTLD_GLOBAL symbol leakage across invocations)."""
    run_dir = os.path.join(repo_scratch_root, "run2")
    argv_orig = sys.argv
    sys.argv = ["gpu_batch_policy_abi_artifact_matrix.py",
               "--scratch-root", run_dir]
    try:
        rc = matrix.main()
    finally:
        sys.argv = argv_orig
    assert rc == 0


@requires_toolchain
def test_task_551_undefined_symbol_is_reproduced_and_fixed(tmp_path):
    build_dir = str(tmp_path / "t551")
    os.makedirs(build_dir, exist_ok=True)
    core_check = matrix.load_core_check()
    _static_r, _lib_a, _obj = matrix.build_static_matrix(build_dir, core_check)
    shared_r, oracle_so, _cc, _cpp = matrix.build_shared_matrix(build_dir, core_check)
    assert shared_r["gate"]
    harness_exe = matrix.build_dlopen_status_harness(build_dir)
    t551 = matrix.build_shared_consumer_pair(build_dir, oracle_so, harness_exe)
    assert t551["broken_link_ok"]
    assert t551["broken_undefined_in_dynsym"]
    assert t551["broken_dlopen_failed_as_expected"]
    assert "lb_gpu_batch_policy_query" in (t551["broken_dlopen_error"] or "")

    # the fixed consumer's own ELF dynamic contract must be exactly right --
    # correct NEEDED/SONAME, and critically NO embedded RPATH/RUNPATH/TEXTREL
    # (task-551 rework: an absolute -Wl,-rpath was itself the violation the
    # manager rejected the prior candidate for). The NEEDED contract is
    # exactly-one-oracle-SONAME + no duplicates + no entry outside the
    # explicit allowed-system set -- NOT exact-list equality against
    # [oracle_soname], which the manager's independent pytest run showed is
    # internally inconsistent with a real, correctly-linked -O0 consumer that
    # legitimately also NEEDs libc.so.6 for its own memcpy/memset calls.
    contract = t551["fixed_dynamic_contract"]
    assert t551["fixed_needed_ok"]
    assert t551["fixed_soname_ok"]
    assert t551["fixed_no_rpath_runpath_ok"]
    assert t551["fixed_no_textrel_ok"]
    assert t551["fixed_external_symbol_undefined_ok"]
    assert contract["needed_oracle_count"] == 1
    assert contract["needed_duplicates"] == []
    assert contract["needed_unexpected_system"] == []
    assert os.path.basename(oracle_so) in contract["needed"]

    # the actual call, run in an isolated dlopen(RTLD_NOW) child process whose
    # only library-search signal is a repo-local LD_LIBRARY_PATH, must both
    # load and return exactly LB_STATUS_OK -- a resolved-but-wrong-status call
    # must not be able to false-green this gate.
    assert t551["fixed_dlopen_ok"]
    assert t551["fixed_call_status"] == 0  # LB_STATUS_OK
    assert t551["gate"]


@requires_toolchain
def test_shared_consumer_dynamic_contract_mutants_fail_closed(tmp_path):
    """Task-551 rework: each dynamic-contract violation class the manager
    flagged as unchecked -- an embedded RPATH, a missing, wrong, or surplus
    NEEDED entry, a NEEDED string-match that still fails to resolve at
    runtime, and a resolved call returning a non-OK status -- must be
    independently caught by the fixed-consumer gate, both directly here and
    inside the full CLI matrix's task551_mutants_ok gate."""
    build_dir = str(tmp_path / "t551_mutants")
    os.makedirs(build_dir, exist_ok=True)
    core_check = matrix.load_core_check()
    _static_r, _lib_a, _obj = matrix.build_static_matrix(build_dir, core_check)
    shared_r, oracle_so, _cc, _cpp = matrix.build_shared_matrix(build_dir, core_check)
    assert shared_r["gate"]
    harness_exe = matrix.build_dlopen_status_harness(build_dir)

    mutants_r = matrix.build_shared_consumer_dynamic_contract_mutants(
        build_dir, oracle_so, harness_exe)
    m = mutants_r["mutants"]

    # (a) embedded RPATH: loads and calls fine (status OK) -- only the
    # dynamic-contract inspection catches it.
    assert m["rpath_injected"]["call_ok"]
    assert m["rpath_injected"]["call_status"] == 0
    assert not m["rpath_injected"]["contract"]["no_rpath_runpath_ok"]
    assert not m["rpath_injected"]["gate"]
    assert m["rpath_injected"]["violation_detected"]

    # (b) missing NEEDED: contract inspection AND the isolated dlopen call
    # both independently show the violation.
    assert not m["missing_needed"]["contract"]["needed_ok"]
    assert not m["missing_needed"]["call_ok"]
    assert not m["missing_needed"]["gate"]
    assert m["missing_needed"]["violation_detected"]

    # (c) wrong NEEDED: a same-ABI decoy that resolves and returns OK, but
    # under the wrong SONAME -- only the NEEDED-string check catches it.
    assert m["wrong_needed"]["call_ok"]
    assert m["wrong_needed"]["call_status"] == 0
    assert not m["wrong_needed"]["contract"]["needed_ok"]
    assert not m["wrong_needed"]["gate"]
    assert m["wrong_needed"]["violation_detected"]

    # (d) unresolved symbol despite a NEEDED entry that string-matches --
    # only the actual isolated dlopen(RTLD_NOW) call catches it.
    assert not m["unresolved_symbol"]["call_ok"]
    assert not m["unresolved_symbol"]["gate"]
    assert m["unresolved_symbol"]["violation_detected"]
    assert "lb_gpu_batch_policy_query" in m["unresolved_symbol"]["harness_output"]

    # (e) nonzero call status: link/load succeed, but the resolved
    # implementation itself returns a non-OK status.
    assert m["nonzero_call_status"]["call_ok"]
    assert m["nonzero_call_status"]["call_status"] != 0
    assert not m["nonzero_call_status"]["gate"]
    assert m["nonzero_call_status"]["violation_detected"]

    # (f) surplus NEEDED: correctly linked against the real oracle (loads and
    # calls fine, status OK) plus one extra, non-allowed NEEDED entry forced
    # in via --no-as-needed -- proves the allowed-system-set replacement for
    # the old exact-list check still rejects a genuine surplus dependency
    # rather than having been loosened to substring/contains matching.
    assert m["surplus_needed"]["call_ok"]
    assert m["surplus_needed"]["call_status"] == 0
    assert m["surplus_needed"]["contract"]["needed_oracle_count"] == 1
    assert m["surplus_needed"]["contract"]["needed_unexpected_system"]
    assert not m["surplus_needed"]["contract"]["needed_ok"]
    assert not m["surplus_needed"]["gate"]
    assert m["surplus_needed"]["violation_detected"]

    assert mutants_r["gate"]


@requires_toolchain
def test_i386_freestanding_eh_frame_has_r386_pc32(tmp_path):
    build_dir = str(tmp_path / "i386")
    os.makedirs(build_dir, exist_ok=True)
    r = matrix.build_i386_matrix(build_dir)
    if not r["i386_toolchain_available"]:
        pytest.skip("cc -m32 i386 codegen unavailable: {}".format(r.get("i386_skip_reason", "")))
    assert r["eh_frame_present_with_unwind"]
    assert r["eh_frame_has_r386_pc32"]
    assert r["eh_frame_absent_without_unwind"]
    assert r["global_no_relocation_assumption_invalidated"]
    # negative_c_freestanding_syntax_ok must be part of the load-bearing gate
    # (see test_i386_matrix_forced_negative_c_compile_failure_fails_gate for
    # the negative direction) -- not merely recorded and ignored.
    assert r["negative_c_freestanding_syntax_ok"]
    assert r["gate"]


@requires_toolchain
def test_i386_matrix_forced_negative_c_compile_failure_fails_gate(tmp_path):
    """Review regression: negative_c_freestanding_syntax_ok was recorded but
    omitted from build_i386_matrix()'s R["gate"], so a broken freestanding
    NEGATIVE_C compile could still leave the i386 gate (and therefore the
    overall CLI gate) PASS. Force the compile to fail via an injected bad
    source and prove both the field and the gate go False together."""
    build_dir = str(tmp_path / "i386_forced_fail")
    os.makedirs(build_dir, exist_ok=True)
    bad_negative_c = os.path.join(build_dir, "broken_negative.c")
    with open(bad_negative_c, "w") as f:
        f.write("this is not valid freestanding C syntax {{{ ???\n")
    r = matrix.build_i386_matrix(build_dir, negative_c_path=bad_negative_c)
    if not r["i386_toolchain_available"]:
        pytest.skip("cc -m32 i386 codegen unavailable: {}".format(
            r.get("i386_skip_reason", "")))
    assert not r["negative_c_freestanding_syntax_ok"]
    assert "negative_c_freestanding_reason" in r
    # every other term the accepted probe already satisfies -- isolates the
    # failure to exactly the negative_c_freestanding_syntax_ok term.
    assert r["elf32_class_ok"]
    assert r["elf32_machine_ok"]
    assert r["global_no_relocation_assumption_invalidated"]
    assert r["eh_frame_absent_without_unwind"]
    assert not r["gate"]


@requires_toolchain
def test_full_matrix_fails_overall_when_negative_c_freestanding_compile_forced_broken(
        monkeypatch, repo_scratch_root):
    """End-to-end companion: forcing the freestanding NEGATIVE_C compile to
    fail through the real CLI main() must flip both i386_ok and **overall**
    to FAIL in the printed gate table and report.md -- proves
    negative_c_freestanding_syntax_ok is load-bearing in the full pipeline,
    not only in a direct build_i386_matrix() call."""
    probe_dir = os.path.join(repo_scratch_root, "i386_probe")
    os.makedirs(probe_dir, exist_ok=True)
    probe = matrix.build_i386_matrix(probe_dir)
    if not probe["i386_toolchain_available"]:
        pytest.skip("cc -m32 i386 codegen unavailable: {}".format(
            probe.get("i386_skip_reason", "")))

    bad_negative_c = os.path.join(repo_scratch_root, "broken_negative.c")
    with open(bad_negative_c, "w") as f:
        f.write("this is not valid freestanding C syntax {{{ ???\n")
    monkeypatch.setattr(matrix, "NEGATIVE_C", bad_negative_c)

    run_dir = os.path.join(repo_scratch_root, "run_forced_fail")
    argv_orig = sys.argv
    sys.argv = ["gpu_batch_policy_abi_artifact_matrix.py",
               "--scratch-root", run_dir]
    try:
        rc = matrix.main()
    finally:
        sys.argv = argv_orig
    assert rc != 0
    report = os.path.join(run_dir, "report.md")
    assert os.path.isfile(report)
    with open(report) as f:
        text = f.read()
    assert "| i386_ok | FAIL |" in text
    assert "**overall** | **FAIL**" in text


# --------------------------------------------------------------------------- #
# _evaluate_archive_contract: pure decision-function coverage of the exact   #
# index/string-table/object presence, order, and deterministic-mode         #
# contract. Needs no toolchain -- constructed member lists mirror exactly   #
# what parse_archive() returns for a genuine `ar rcsD` archive.             #
# --------------------------------------------------------------------------- #
def _member(name, mode, mtime=b"0", uid=b"0", gid=b"0", size=0, is_index=None):
    if is_index is None:
        is_index = name in ("/", "//")
    return {"name": name, "mode": mode, "mtime": mtime, "uid": uid,
            "gid": gid, "size": size, "is_index": is_index}


def _genuine_members(object_size=1280):
    # Mirrors an actual `ar rcsD` output observed on this toolchain: index
    # mode "0", string-table mode blank, real object mode "644".
    return [
        _member("/", b"0"),
        _member("//", b""),
        _member("gpu_batch_policy_abi_oracle.o", b"644", size=object_size),
    ]


def test_evaluate_archive_contract_accepts_genuine_order_and_modes():
    r = matrix._evaluate_archive_contract(
        _genuine_members(1280), "gpu_batch_policy_abi_oracle.o", 1280)
    assert r["ok"]
    assert r["order_ok"]
    assert r["index_meta_ok"]
    assert r["strtab_meta_ok"]
    assert r["object_meta_ok"]
    assert r["object_identity_ok"]


def test_evaluate_archive_contract_rejects_missing_index_member():
    members = [m for m in _genuine_members() if m["name"] != "/"]
    r = matrix._evaluate_archive_contract(
        members, "gpu_batch_policy_abi_oracle.o", 1280)
    assert not r["ok"]
    assert not r["order_ok"]
    assert not r["count_ok"]


def test_evaluate_archive_contract_rejects_reordered_members():
    members = list(reversed(_genuine_members()))
    r = matrix._evaluate_archive_contract(
        members, "gpu_batch_policy_abi_oracle.o", 1280)
    assert not r["ok"]
    assert not r["order_ok"]


def test_evaluate_archive_contract_rejects_altered_object_mode():
    members = _genuine_members()
    members[2] = dict(members[2], mode=b"755")
    r = matrix._evaluate_archive_contract(
        members, "gpu_batch_policy_abi_oracle.o", 1280)
    assert not r["ok"]
    assert r["order_ok"]  # order is still correct -- isolates the mode term
    assert not r["object_meta_ok"]


def test_evaluate_archive_contract_rejects_altered_index_mode():
    members = _genuine_members()
    members[0] = dict(members[0], mode=b"644")
    r = matrix._evaluate_archive_contract(
        members, "gpu_batch_policy_abi_oracle.o", 1280)
    assert not r["ok"]
    assert not r["index_meta_ok"]


def test_evaluate_archive_contract_rejects_nonzero_mtime_uid_gid():
    members = _genuine_members()
    members[2] = dict(members[2], mtime=b"1700000000")
    r = matrix._evaluate_archive_contract(
        members, "gpu_batch_policy_abi_oracle.o", 1280)
    assert not r["ok"]
    assert not r["object_meta_ok"]


def test_evaluate_archive_contract_rejects_wrong_object_size():
    r = matrix._evaluate_archive_contract(
        _genuine_members(1280), "gpu_batch_policy_abi_oracle.o", 999)
    assert not r["ok"]
    assert not r["object_identity_ok"]


def test_evaluate_archive_contract_rejects_wrong_object_name():
    r = matrix._evaluate_archive_contract(
        _genuine_members(1280), "some_other_object.o", 1280)
    assert not r["ok"]
    assert not r["order_ok"]


@requires_toolchain
def test_static_archive_matrix_has_exact_genuine_order_and_modes(tmp_path):
    """Real-build companion to the pure-function tests above: build_static_
    matrix()'s own freshly built `ar rcsD` archive must satisfy the full
    contract, and archive_order_ok/archive_deterministic_ok must be part of
    R["gate"] (not merely recorded)."""
    build_dir = str(tmp_path / "static_order")
    os.makedirs(build_dir, exist_ok=True)
    core_check = matrix.load_core_check()
    r, _lib_a, _obj = matrix.build_static_matrix(build_dir, core_check)
    assert r["archive_order_ok"]
    assert r["archive_contract"]["ok"]
    assert r["archive_deterministic_ok"]
    assert r["archive_real_member_ok"]
    assert r["gate"]


@requires_toolchain
def test_static_archive_contract_mutants_fail_closed(tmp_path):
    """Task-561 rework: missing symbol index, reordered index/string-table/
    object members, and an altered real-member mode must each be
    independently caught by _evaluate_archive_contract() when applied to a
    genuine, byte-mutated `ar rcsD` archive -- not fabricated/synthetic
    archive bytes."""
    build_dir = str(tmp_path / "static_mutants")
    os.makedirs(build_dir, exist_ok=True)
    core_check = matrix.load_core_check()
    _r, lib_a, obj = matrix.build_static_matrix(build_dir, core_check)

    mutants_r = matrix.build_static_archive_contract_mutants(
        build_dir, lib_a, obj)
    m = mutants_r["mutants"]

    assert not m["missing_index"]["ok"]
    assert m["missing_index"]["violation_detected"]
    assert not m["missing_index"]["count_ok"]

    assert not m["reordered_members"]["ok"]
    assert m["reordered_members"]["violation_detected"]
    assert not m["reordered_members"]["order_ok"]

    assert not m["altered_mode"]["ok"]
    assert m["altered_mode"]["violation_detected"]
    assert m["altered_mode"]["order_ok"]  # order intact; only mode is wrong
    assert not m["altered_mode"]["object_meta_ok"]

    assert mutants_r["gate"]


# --------------------------------------------------------------------------- #
# _child_env_for_dlopen / _run_dlopen_harness: the isolated dlopen(RTLD_NOW) #
# child must never inherit or append this process's own LD_LIBRARY_PATH (or  #
# any other environment variable) -- only the caller-validated repo-local    #
# directory may reach it.                                                    #
# --------------------------------------------------------------------------- #
def test_child_env_for_dlopen_ignores_hostile_parent_ld_library_path(monkeypatch):
    """The child's LD_LIBRARY_PATH must be exactly the validated directory,
    never the parent's value and never the parent's value with the
    validated directory merely prepended/appended to it."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/hostile_attacker_controlled_dir")
    env = matrix._child_env_for_dlopen("/repo/local/validated/dir")
    assert env["LD_LIBRARY_PATH"] == "/repo/local/validated/dir"
    assert "/tmp/hostile_attacker_controlled_dir" not in env["LD_LIBRARY_PATH"]


def test_child_env_for_dlopen_sets_ld_library_path_when_parent_unset(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    env = matrix._child_env_for_dlopen("/repo/local/validated/dir")
    assert env["LD_LIBRARY_PATH"] == "/repo/local/validated/dir"


def test_child_env_for_dlopen_preserves_other_environment_variables(monkeypatch):
    """Only LD_LIBRARY_PATH is overridden -- this is not a general
    environment-stripping sandbox, so unrelated variables the calling
    process depends on (e.g. PATH) must still reach the child."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/hostile_attacker_controlled_dir")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "keep_me")
    env = matrix._child_env_for_dlopen("/repo/local/validated/dir")
    assert env.get("SOME_UNRELATED_VAR") == "keep_me"
    assert env.get("PATH") == os.environ.get("PATH")


@requires_toolchain
def test_dlopen_harness_subprocess_env_excludes_hostile_parent_ld_library_path(
        tmp_path, monkeypatch):
    """Integration-level proof: a real _run_dlopen_harness() call, against a
    real compiled fixed consumer, must pass exactly {"LD_LIBRARY_PATH":
    build_dir} to subprocess.run -- a hostile LD_LIBRARY_PATH set in this
    test process's own environment must not appear in it, and the real call
    must still succeed and return LB_STATUS_OK using only the validated
    directory."""
    build_dir = str(tmp_path / "hostile_env_e2e")
    os.makedirs(build_dir, exist_ok=True)
    core_check = matrix.load_core_check()
    _static_r, _lib_a, _obj = matrix.build_static_matrix(build_dir, core_check)
    shared_r, oracle_so, _cc, _cpp = matrix.build_shared_matrix(build_dir, core_check)
    assert shared_r["gate"]
    harness_exe = matrix.build_dlopen_status_harness(build_dir)
    t551 = matrix.build_shared_consumer_pair(build_dir, oracle_so, harness_exe)
    assert t551["gate"]
    fixed = os.path.join(build_dir, matrix.FIXED_CONSUMER_SONAME)

    captured = {}
    real_run = matrix.subprocess.run

    def spy_run(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(matrix.subprocess, "run", spy_run)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/nonexistent/hostile/attacker/path")

    call_ok, status, _out = matrix._run_dlopen_harness(harness_exe, fixed, build_dir)

    assert call_ok
    assert status == 0
    assert captured["env"]["LD_LIBRARY_PATH"] == build_dir
    assert "/nonexistent/hostile/attacker/path" not in captured["env"]["LD_LIBRARY_PATH"]


# --------------------------------------------------------------------------- #
# Review rework (HIGH): _child_env_for_dlopen previously stripped only       #
# LD_LIBRARY_PATH, leaving LD_PRELOAD/LD_AUDIT (and every other glibc        #
# loader-control variable) inherited from this process's own environment --  #
# a hostile parent could define or intercept lb_gpu_batch_policy_query       #
# itself, independent of the artifact's own correct dynamic contract.        #
# --------------------------------------------------------------------------- #
def test_child_env_for_dlopen_strips_ld_preload_and_ld_audit(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/tmp/hostile_attacker_controlled.so")
    monkeypatch.setenv("LD_AUDIT", "/tmp/hostile_attacker_audit.so")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "keep_me")
    env = matrix._child_env_for_dlopen("/repo/local/validated/dir")
    assert "LD_PRELOAD" not in env
    assert "LD_AUDIT" not in env
    assert env["LD_LIBRARY_PATH"] == "/repo/local/validated/dir"
    assert env.get("SOME_UNRELATED_VAR") == "keep_me"
    assert env.get("PATH") == os.environ.get("PATH")


def test_child_env_for_dlopen_strips_full_loader_control_variable_set(monkeypatch):
    for name in matrix._LOADER_CONTROL_ENV_VARS:
        monkeypatch.setenv(name, "/tmp/hostile_value_for_" + name)
    env = matrix._child_env_for_dlopen("/repo/local/validated/dir")
    for name in matrix._LOADER_CONTROL_ENV_VARS:
        if name == "LD_LIBRARY_PATH":
            continue
        assert name not in env, f"{name} must be stripped, not just LD_LIBRARY_PATH"
    assert env["LD_LIBRARY_PATH"] == "/repo/local/validated/dir"


@requires_toolchain
def test_dlopen_harness_subprocess_env_excludes_hostile_ld_preload_and_ld_audit(
        tmp_path, monkeypatch):
    """Integration-level companion to the unit tests above: a real
    _run_dlopen_harness() call must pass an env to subprocess.run that
    contains neither LD_PRELOAD nor LD_AUDIT even when both are set (to
    real, existing files) in this test process's own environment."""
    build_dir = str(tmp_path / "hostile_loader_vars_e2e")
    os.makedirs(build_dir, exist_ok=True)
    core_check = matrix.load_core_check()
    _static_r, _lib_a, _obj = matrix.build_static_matrix(build_dir, core_check)
    shared_r, oracle_so, _cc, _cpp = matrix.build_shared_matrix(build_dir, core_check)
    assert shared_r["gate"]
    harness_exe = matrix.build_dlopen_status_harness(build_dir)
    t551 = matrix.build_shared_consumer_pair(build_dir, oracle_so, harness_exe)
    assert t551["gate"]
    fixed = os.path.join(build_dir, matrix.FIXED_CONSUMER_SONAME)

    captured = {}
    real_run = matrix.subprocess.run

    def spy_run(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(matrix.subprocess, "run", spy_run)
    monkeypatch.setenv("LD_PRELOAD", harness_exe)  # any real, existing file
    monkeypatch.setenv("LD_AUDIT", harness_exe)

    call_ok, status, _out = matrix._run_dlopen_harness(harness_exe, fixed, build_dir)

    assert call_ok
    assert status == 0
    assert "LD_PRELOAD" not in captured["env"]
    assert "LD_AUDIT" not in captured["env"]


@requires_toolchain
def test_hostile_ld_preload_and_ld_audit_cannot_hijack_task551_fixed_consumer(
        tmp_path, monkeypatch):
    """Task-551 rework: a REAL hostile shared library that defines
    lb_gpu_batch_policy_query itself (returning a distinctive non-OK status
    a hijack would leak) is put on LD_PRELOAD/LD_AUDIT in this process's own
    environment. Proves two things with a genuine, load-bearing mutant, not
    a synthetic/inert one:

      (a) the SAME hostile library, run through a naively-stripped
          environment that (like the pre-rework code) only removes
          LD_LIBRARY_PATH, really does hijack the call -- so this is a real
          exploitable mechanism, not a no-op;
      (b) _run_dlopen_harness()'s actual environment construction makes that
          same hijack attempt resolve to nothing but the real oracle
          (status LB_STATUS_OK, never the hostile 424242), and the
          undefined-symbol broken-consumer reproduction (an in-process
          ctypes.CDLL() call, never a subprocess -- LD_PRELOAD/LD_AUDIT are
          only consulted by the dynamic loader at process startup, so
          mutating this already-running process's environment cannot affect
          it either way) still fails exactly as task 551 originally did."""
    build_dir = str(tmp_path / "hostile_hijack_e2e")
    os.makedirs(build_dir, exist_ok=True)
    core_check = matrix.load_core_check()
    _static_r, _lib_a, _obj = matrix.build_static_matrix(build_dir, core_check)
    shared_r, oracle_so, _cc, _cpp = matrix.build_shared_matrix(build_dir, core_check)
    assert shared_r["gate"]
    harness_exe = matrix.build_dlopen_status_harness(build_dir)
    t551 = matrix.build_shared_consumer_pair(build_dir, oracle_so, harness_exe)
    assert t551["gate"]

    hijack_src = os.path.join(build_dir, "hostile_ld_preload.c")
    with open(hijack_src, "w") as f:
        f.write(
            '#include "gpu_batch_policy_abi_contract.h"\n'
            "lb_status lb_gpu_batch_policy_query(const void *q, size_t qs,\n"
            "                                    void *r, size_t rs) {\n"
            "    (void)q; (void)qs; (void)r; (void)rs;\n"
            "    return (lb_status)424242;\n"
            "}\n")
    hijack_lib = os.path.join(build_dir, "hostile_ld_preload.so")
    matrix.compile_cmd([matrix.CC, "-std=c11", "-O0", "-fPIC", "-shared",
                        "-I", matrix.HERE, hijack_src, "-o", hijack_lib])
    fixed = os.path.join(build_dir, matrix.FIXED_CONSUMER_SONAME)

    # (a) precondition: prove the hostile mutant is real by reproducing the
    # pre-rework (LD_LIBRARY_PATH-only-stripped) vulnerable behavior.
    naive_env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    naive_env["LD_LIBRARY_PATH"] = build_dir
    naive_env["LD_PRELOAD"] = hijack_lib
    p = subprocess.run([harness_exe, fixed], env=naive_env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       check=False)
    assert b"STATUS 424242" in p.stdout, (
        "hostile LD_PRELOAD mutant did not hijack the naive environment -- "
        "test precondition invalid, this is not a real exploit mutant")

    # (b) the fix: with the hostile library on LD_PRELOAD and LD_AUDIT in
    # this process's own environment, the isolated harness call must still
    # resolve only the real oracle.
    monkeypatch.setenv("LD_PRELOAD", hijack_lib)
    monkeypatch.setenv("LD_AUDIT", hijack_lib)
    call_ok, status, _out = matrix._run_dlopen_harness(harness_exe, fixed, build_dir)
    assert call_ok
    assert status == 0  # LB_STATUS_OK from the real oracle, never 424242

    # the broken-consumer reproduction, re-run with the hostile env still
    # set on this process, must still fail exactly as before.
    t551_hostile = matrix.build_shared_consumer_pair(build_dir, oracle_so, harness_exe)
    assert t551_hostile["broken_link_ok"]
    assert t551_hostile["broken_undefined_in_dynsym"]
    assert t551_hostile["broken_dlopen_failed_as_expected"]
    assert t551_hostile["gate"]


# --------------------------------------------------------------------------- #
# Review rework (LOW): _run_dlopen_harness()'s audit-trail command string     #
# previously read "env -i LD_LIBRARY_PATH=... harness lib" -- `env -i` means  #
# a completely empty environment, which misrepresents what                    #
# _child_env_for_dlopen() actually does (strip only the loader-control        #
# variables in _LOADER_CONTROL_ENV_VARS; PATH and every other non-loader      #
# variable is passed through unchanged, per                                   #
# test_child_env_for_dlopen_preserves_other_environment_variables above).     #
# report.md security evidence built from that string overstated the          #
# isolation guarantee actually enforced.                                      #
# --------------------------------------------------------------------------- #
def test_dlopen_harness_command_repr_does_not_claim_env_dash_i():
    """The audit-trail command representation must not use `env -i` (empty
    environment) notation -- that is a stronger, false isolation claim than
    the loader-control-only sanitization _child_env_for_dlopen() performs."""
    repr_str = matrix._dlopen_harness_command_repr(
        "/build/dlopen_status_harness", "/build/libfixed.so.1", "/build")
    assert "env -i" not in repr_str
    assert "-i " not in repr_str.split("LD_LIBRARY_PATH", 1)[0]


def test_dlopen_harness_command_repr_labels_loader_sanitization_precisely():
    """The representation must name the actually-executed environment
    policy (loader-control variables sanitized) and the exact
    LD_LIBRARY_PATH/executable/argument that were passed -- deterministic,
    non-shell notation, not a claim of full environment clearing."""
    repr_str = matrix._dlopen_harness_command_repr(
        "/build/dlopen_status_harness", "/build/libfixed.so.1", "/build")
    assert repr_str == (
        "env [loader-controls-sanitized] LD_LIBRARY_PATH=/build "
        "/build/dlopen_status_harness /build/libfixed.so.1")


def test_dlopen_harness_command_repr_does_not_serialize_inherited_env(monkeypatch):
    """The representation must never dump this process's inherited
    environment (which could contain secrets) -- only the one explicitly
    constructed LD_LIBRARY_PATH value may appear."""
    monkeypatch.setenv("SOME_SECRET_LOOKING_VAR", "super-secret-token-value")
    repr_str = matrix._dlopen_harness_command_repr(
        "/build/dlopen_status_harness", "/build/libfixed.so.1", "/build")
    assert "SOME_SECRET_LOOKING_VAR" not in repr_str
    assert "super-secret-token-value" not in repr_str


@requires_toolchain
def test_run_dlopen_harness_records_loader_sanitized_command_for_real_call(
        tmp_path, monkeypatch):
    """Integration-level companion: a real _run_dlopen_harness() call must
    append the loader-controls-sanitized command representation (not
    `env -i`) to the module's _commands audit trail, even while non-loader
    variables (PATH) genuinely reach the child (test_child_env_for_dlopen_
    preserves_other_environment_variables above proves the child actually
    receives them) -- so the report.md evidence and the real executed
    policy stay in agreement."""
    build_dir = str(tmp_path / "command_repr_e2e")
    os.makedirs(build_dir, exist_ok=True)
    core_check = matrix.load_core_check()
    _static_r, _lib_a, _obj = matrix.build_static_matrix(build_dir, core_check)
    shared_r, oracle_so, _cc, _cpp = matrix.build_shared_matrix(build_dir, core_check)
    assert shared_r["gate"]
    harness_exe = matrix.build_dlopen_status_harness(build_dir)
    t551 = matrix.build_shared_consumer_pair(build_dir, oracle_so, harness_exe)
    assert t551["gate"]
    fixed = os.path.join(build_dir, matrix.FIXED_CONSUMER_SONAME)

    before_len = len(matrix._commands)
    call_ok, status, _out = matrix._run_dlopen_harness(harness_exe, fixed, build_dir)
    assert call_ok
    assert status == 0
    assert len(matrix._commands) == before_len + 1
    recorded = matrix._commands[-1]
    assert "env -i" not in recorded
    assert recorded == matrix._dlopen_harness_command_repr(
        harness_exe, fixed, build_dir)


# --------------------------------------------------------------------------- #
# _validate_scratch_root: fail-closed --scratch-root validation. Every       #
# rejection case must perform no write; only a path strictly beneath the     #
# authorized repo out/ tree, with no symlink component, is accepted.         #
# --------------------------------------------------------------------------- #
def test_validate_scratch_root_rejects_path_outside_out_tree():
    """The rejected candidate must be deterministic and lexically inside this
    repository but strictly outside the canonical out/ tree -- not derived
    from pytest's tmp_path, whose --basetemp may itself be placed beneath
    the authorized out/ tree, which would make it wrongly accepted."""
    outside = os.path.join(REPO, "definitely_outside_repo_out_tree_scratch_probe")
    assert not os.path.exists(outside)
    with pytest.raises(SystemExit):
        matrix._validate_scratch_root(outside)
    assert not os.path.exists(outside)


def test_validate_scratch_root_rejects_out_tree_root_itself():
    with pytest.raises(SystemExit):
        matrix._validate_scratch_root(matrix.AUTHORIZED_SCRATCH_PARENT)


def test_validate_scratch_root_rejects_escape_via_dotdot():
    escape = os.path.join(matrix.AUTHORIZED_SCRATCH_PARENT, "..",
                          "escaped_via_dotdot_outside_out_tree")
    with pytest.raises(SystemExit):
        matrix._validate_scratch_root(escape)
    assert not os.path.exists(os.path.join(REPO, "escaped_via_dotdot_outside_out_tree"))


def test_validate_scratch_root_accepts_path_strictly_inside_out_tree():
    inside = os.path.join(
        matrix.AUTHORIZED_SCRATCH_PARENT, ".tasks", matrix.TASK_ID,
        "pytest_validate_accept_test")
    result = matrix._validate_scratch_root(inside)
    assert result == os.path.realpath(inside)
    # validation alone must perform no write.
    assert not os.path.exists(inside)


def test_validate_scratch_root_rejects_symlink_component_even_when_target_is_inside_out_tree():
    """A symlink component must be rejected even when its resolved target
    would otherwise land inside the authorized out/ tree -- proves the
    symlink check is independent of (and enforced before) the containment
    check, not merely redundant with it, and that rejection happens with no
    write performed through the symlink."""
    base = os.path.join(matrix.AUTHORIZED_SCRATCH_PARENT, ".tasks",
                        matrix.TASK_ID, "pytest_symlink_component_test")
    real_target = os.path.join(base, "real_target")
    symlink_path = os.path.join(base, "via_symlink")
    os.makedirs(real_target, exist_ok=True)
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(real_target, symlink_path)
    try:
        candidate = os.path.join(symlink_path, "scratch")
        with pytest.raises(SystemExit):
            matrix._validate_scratch_root(candidate)
        assert not os.path.exists(os.path.join(real_target, "scratch"))
    finally:
        os.remove(symlink_path)
        shutil.rmtree(real_target, ignore_errors=True)


@requires_toolchain
def test_main_rejects_scratch_root_outside_out_tree_before_any_write(monkeypatch):
    """End-to-end: main() must reject an external --scratch-root before
    os.makedirs() or any build step runs. The rejected candidate is a
    deterministic, repo-local path outside the canonical out/ tree -- not
    derived from pytest's tmp_path, whose --basetemp may itself land beneath
    the authorized out/ tree. os.makedirs is monkeypatched to record any call
    and raise immediately, so a regression that reaches it cannot silently
    create junk and is caught as a distinct failure rather than merely
    inferred from pytest.raises(SystemExit) succeeding for the wrong reason."""
    outside = os.path.join(
        REPO, "external_scratch_root_via_cli_outside_out_tree_probe")
    assert not os.path.exists(outside)

    makedirs_calls = []

    def spy_makedirs(*args, **kwargs):
        makedirs_calls.append((args, kwargs))
        raise AssertionError(
            "os.makedirs must not be reached: --scratch-root validation "
            "must reject before any write")

    monkeypatch.setattr(matrix.os, "makedirs", spy_makedirs)

    argv_orig = sys.argv
    sys.argv = ["gpu_batch_policy_abi_artifact_matrix.py",
               "--scratch-root", outside]
    try:
        with pytest.raises(SystemExit):
            matrix.main()
    finally:
        sys.argv = argv_orig

    assert makedirs_calls == []
    assert not os.path.exists(outside)


# --------------------------------------------------------------------------- #
# Review rework (MEDIUM): _validate_scratch_root() only checked the          #
# candidate path once and returned a plain string; os.makedirs() and every   #
# subsequent os.path.join(build_dir, ...) write resolved that same mutable   #
# pathname a second (and third, and fourth, ...) time, so a symlink/rename   #
# substitution occurring strictly after validation could redirect writes     #
# outside the authorized out/ tree. _pin_scratch_dir() now re-derives the    #
# validated path as a chain of O_NOFOLLOW-opened directory file descriptors  #
# and returns a "/proc/<pid>/fd/<fd>" pinned path every subsequent write     #
# joins onto -- immune to any later rename/symlink substitution of an        #
# ancestor pathname component.                                               #
# --------------------------------------------------------------------------- #
def test_pin_scratch_dir_matches_validated_path_and_is_writable():
    target = os.path.join(
        matrix.AUTHORIZED_SCRATCH_PARENT, ".tasks", matrix.TASK_ID,
        "pytest_pin_basic_test")
    shutil.rmtree(target, ignore_errors=True)
    validated = matrix._validate_scratch_root(target)
    fd, pinned_path = matrix._pin_scratch_dir(validated)
    try:
        assert pinned_path == f"/proc/{os.getpid()}/fd/{fd}"
        with open(os.path.join(pinned_path, "probe.txt"), "w") as f:
            f.write("pinned")
        assert os.path.isfile(os.path.join(target, "probe.txt"))
    finally:
        os.close(fd)
        shutil.rmtree(target, ignore_errors=True)


def test_pin_scratch_dir_write_survives_post_pin_ancestor_symlink_substitution():
    """The core TOCTOU regression: after _pin_scratch_dir() has already
    pinned the scratch directory, an attacker renames the real directory
    aside and replaces its name with a symlink to a decoy directory. A write
    performed afterward through the pinned fd/path must still land in the
    ORIGINAL directory (now at its renamed-aside path) and must never reach
    the attacker's decoy -- proving the pin is immune to a substitution that
    happens strictly after validation, which a plain resolved-path string
    (the pre-rework behavior) cannot resist."""
    base = os.path.join(matrix.AUTHORIZED_SCRATCH_PARENT, ".tasks",
                        matrix.TASK_ID, "pytest_pin_substitution_test")
    real_target = os.path.join(base, "victim")
    moved_aside = real_target + "_moved_by_attacker"
    decoy = os.path.join(base, "decoy_outside_target")
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)

    validated = matrix._validate_scratch_root(real_target)
    fd, pinned_path = matrix._pin_scratch_dir(validated)
    try:
        os.makedirs(decoy, exist_ok=True)
        os.rename(real_target, moved_aside)
        os.symlink(decoy, real_target)

        with open(os.path.join(pinned_path, "proof_of_pin.txt"), "w") as f:
            f.write("pinned")

        assert os.path.isfile(os.path.join(moved_aside, "proof_of_pin.txt"))
        assert not os.path.exists(os.path.join(decoy, "proof_of_pin.txt"))
    finally:
        os.close(fd)
        if os.path.islink(real_target):
            os.remove(real_target)
        shutil.rmtree(base, ignore_errors=True)


def test_pin_scratch_dir_rejects_out_tree_root_itself():
    with pytest.raises(SystemExit):
        matrix._pin_scratch_dir(matrix.AUTHORIZED_SCRATCH_PARENT)


@requires_toolchain
def test_main_writes_report_through_pinned_scratch_dir(repo_scratch_root):
    """End-to-end: report.md written by a real main() run through the
    pinned "/proc/<pid>/fd/<fd>" build_dir must be visible at the original,
    real scratch-root pathname (they are the same on-disk directory, just
    reached via a different, TOCTOU-safe route)."""
    run_dir = os.path.join(repo_scratch_root, "pin_e2e_run")
    argv_orig = sys.argv
    sys.argv = ["gpu_batch_policy_abi_artifact_matrix.py",
               "--scratch-root", run_dir]
    try:
        rc = matrix.main()
    finally:
        sys.argv = argv_orig
    assert rc == 0
    assert os.path.isfile(os.path.join(run_dir, "report.md"))


# --------------------------------------------------------------------------- #
# Review rework (MEDIUM): the module-global _commands list was never        #
# cleared, so a second in-process main() call accumulated the first run's    #
# commands on top of its own, making the second run's report.md "Commands    #
# executed (N)" count and command list wrong. main() now clears _commands    #
# (and _RESOLVED_DEPS) as its first action.                                  #
# --------------------------------------------------------------------------- #
@requires_toolchain
def test_main_command_audit_does_not_accumulate_across_repeated_invocations(
        repo_scratch_root):
    """Two independent, complete in-process main() calls must each report
    their own command count/list -- not the first run's commands plus the
    second's."""
    def _command_section(run_dir):
        argv_orig = sys.argv
        sys.argv = ["gpu_batch_policy_abi_artifact_matrix.py",
                   "--scratch-root", run_dir]
        try:
            rc = matrix.main()
        finally:
            sys.argv = argv_orig
        assert rc == 0
        with open(os.path.join(run_dir, "report.md")) as f:
            text = f.read()
        marker = "## 8. Commands executed ("
        start = text.index(marker) + len(marker)
        count = int(text[start:text.index(")", start)])
        cmd_block_start = text.index("```\n", start) + len("```\n")
        cmd_block_end = text.index("\n```", cmd_block_start)
        commands = text[cmd_block_start:cmd_block_end].splitlines()
        assert len(commands) == count
        return count, commands

    run1_dir = os.path.join(repo_scratch_root, "cmd_audit_run1")
    run2_dir = os.path.join(repo_scratch_root, "cmd_audit_run2")
    count1, commands1 = _command_section(run1_dir)
    count2, commands2 = _command_section(run2_dir)

    assert count1 > 0
    # identical operations each run -> identical command counts; a
    # regression that fails to reset _commands would make run2's count
    # roughly double run1's (accumulated on top).
    assert count2 == count1
    # run2's command list must reference only its own build_dir, never
    # run1's -- direct proof no run1 command entries leaked into run2.
    assert not any(run1_dir in c for c in commands2)
    # symmetric proof in the other direction: run1's own command list must
    # never reference run2's build_dir either, confirming commands1 is a
    # genuine, independent snapshot and not aliased/mutated by the second
    # main() call.
    assert not any(run2_dir in c for c in commands1)


def test_commands_module_list_is_cleared_at_start_of_main(repo_scratch_root,
                                                           monkeypatch):
    """Narrower unit-level companion: seed the module-global _commands list
    with stale entries from a hypothetical prior run before calling main(),
    and confirm none of those stale entries survive into the fresh run's
    report.md -- isolates the fix to main()'s own reset step rather than
    relying only on the end-to-end count comparison above."""
    if not TOOLCHAIN_AVAILABLE:
        pytest.skip("cc/c++/ar not all available on PATH")
    matrix._commands.append("STALE_COMMAND_FROM_A_PRIOR_RUN_lb_gpu_batch_policy_query")
    run_dir = os.path.join(repo_scratch_root, "cmd_reset_run")
    argv_orig = sys.argv
    sys.argv = ["gpu_batch_policy_abi_artifact_matrix.py",
               "--scratch-root", run_dir]
    try:
        rc = matrix.main()
    finally:
        sys.argv = argv_orig
    assert rc == 0
    with open(os.path.join(run_dir, "report.md")) as f:
        text = f.read()
    assert "STALE_COMMAND_FROM_A_PRIOR_RUN" not in text
