#!/usr/bin/env python3
"""check_required_checks_match_jobs.py — CAAS-CI-001 regression gate.

Branch-protection required status checks (ci/update_required_checks.sh) must
reference REAL job display names that actually run on pull_request → main.
A required context that matches no job is either ignored (the security gate is
then NOT required at the merge boundary) or, under strict mode, blocks every
merge forever.

This gate caught CAAS-CI-001: the contexts list pinned "linux (gcc-13, …)"
(ci.yml's matrix is gcc-14) and contained ZERO gate.yml CAAS jobs, so the CAAS
security pipeline was not required to merge.

For every context in update_required_checks.sh this gate asserts:
  1. it resolves to a job display name in some .github/workflows/*.yml
     (matrix display names are expanded), AND
  2. that workflow triggers on `pull_request` (else strict mode blocks forever).

This gate also runs a second, independent check on gate.yml specifically:
`check_gpu_export_closure_skip_guard()` structurally validates the issue #335
round-13 contract — that the `gpu-export-closure` job exists, that
`final-verdict` depends on it via `needs:`, that its own `if:` guard is
docs_only-scoped AND correctly worded (tests docs_only for INEQUALITY to
'true', not equality), and that `final-verdict`'s script has an explicit
docs_only-gated pre-check — also direction-checked, and pinned to the exact
`needs.detect-impact.outputs.docs_only` context path rather than any token
merely containing the substring "docs_only" — rejecting a silent skip of
that job (mirroring the pre-existing `caas-security` skip-guard) without
regressing the generic failure/cancelled blocking loop.

Round-13 fix-iteration 2: an independent verifier proved that fix-iteration
1's block-nesting-aware rewrite still validated only TOKEN PRESENCE inside a
correctly-nested if-block, never the comparison OPERATOR/DIRECTION or that
the referenced variable was the REAL `needs.detect-impact.outputs.docs_only`
path. Flipping a single `!=`/`=`/`==`, or swapping in an unrelated
lookalike variable name, produced a script that would NOT actually fail
closed at runtime for a non-docs skip, yet the validator still reported zero
violations. `_equality_test()` below closes this: it requires the anchor
variable and the comparison operator/literal to be directly adjacent in the
condition text (not just co-occurring anywhere in the line), and returns the
comparison's direction so callers can reject an inverted guard, not merely
detect a missing one.

Round 14: an independent round-13 re-verification pass proved
`_equality_test()`'s adjacency search still accepted an otherwise
correctly-worded, correctly-directed comparison with an always-false extra
`&&`/`-a` conjunct appended (real bash's `&&` makes the whole condition
false, permanently dead-coding the guard, while the old window-search still
found the required comparison sitting next to its anchor). `_equality_test()`
now first reduces its input to a single top-level active boolean term via
`_reduce_to_single_active_term()` and rejects outright if more than one term
is present at any nesting depth. This round's own adversarial
self-verification then found a second, structurally different way to
achieve the same "guard looks present but is dead at runtime" outcome:
wrapping the ENTIRE guard, or just its inner docs_only pre-check, inside an
unrelated always-false ANCESTOR if-block (e.g. `if [ "1" = "0" ]; then
<correct guard> fi`) — the then-current `_has_gpu_export_closure_skip_guard()`
searched for its anchors at any nesting depth via a recursive
`_walk_all_if_blocks()` helper, so it still found the correctly-worded
comparison sitting inside that dead ancestor and reported the guard
present. Fixed by restricting both the outer and inner searches to
TOP-LEVEL if-blocks only (`_iter_if_blocks()`, no recursion) — matching how
the real gate.yml has always been structured — plus stripping any further
nested if-block out of the matched inner block's body
(`_strip_nested_if_blocks()`) before searching it for `exit 1`, closing the
same trick one level deeper (wrapping just the `exit 1` itself). Part 3
(same round) widened the recognized block openers/closers beyond `if`/`fi`
to `case`/`esac` and `for`/`while`/`until`/`do`/`done` (`_is_block_open()` /
`_is_block_close()`) — the identical always-dead-wrapper trick spelled with
a never-matching `case` arm or a zero-iteration loop instead of an `if` was
otherwise invisible to an if/fi-only tracker. Part 4 (same round) added a
behavioral defense-in-depth layer, `_gpu_export_closure_guard_blocks_at_runtime()`
— no finite enumeration of block keywords can prove an arbitrary shell
fragment's `exit 1` is unconditionally reached, so once every structural
check passes, this renders the script for the round-12 danger scenario and
executes it via a real `bash -c` subprocess, requiring a non-zero exit;
closes short-circuited `<test> && exit 1` / `<test> || exit 1` statements
(no block keyword at all) and any future shape the static scan does not yet
enumerate. See each function's own docstring and `docs/AUDIT_CHANGELOG.md`'s
"round 14, parts 1-4" entry for full detail.

Exit 0 = every context resolves to a PR-triggered job, 1 = mismatch.
"""
import itertools
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    # The fast-gates job runs under setup-python without PyYAML installed. Parsing
    # GitHub Actions YAML reliably needs a real YAML parser; a hand-rolled one would
    # risk FALSE failures. This gate's job is to catch branch-protection drift, which
    # is introduced ONLY by editing ci/update_required_checks.sh — a change a local
    # pre-push run (PyYAML present) validates before it is ever pushed. So skip loudly
    # rather than crash when PyYAML is unavailable.
    print("::notice::check_required_checks_match_jobs: PyYAML unavailable in this "
          "environment — skipped here. Runs in local pre-push and any PyYAML-equipped "
          "CI job; required-checks drift is caught there before it lands.")
    sys.exit(0)

WORKFLOWS = Path(".github/workflows")
REQUIRED_SH = Path("ci/update_required_checks.sh")


def fail(msg: str) -> None:
    print(f"::error::check_required_checks_match_jobs: {msg}")


def parse_contexts(text: str):
    """Extract the JSON-ish contexts array from the heredoc in the shell script."""
    m = re.search(r'"contexts"\s*:\s*\[(.*?)\]', text, re.S)
    if not m:
        return None
    return re.findall(r'"([^"]+)"', m.group(1))


def matrix_combos(matrix: dict):
    """Yield dicts of matrix var→value, expanding the cartesian product of
    list-valued axes and merging `include` entries (mirrors GitHub Actions)."""
    if not isinstance(matrix, dict):
        return []
    cart_keys = [k for k, v in matrix.items()
                 if k not in ("include", "exclude") and isinstance(v, list)]
    base = [dict(zip(cart_keys, vals))
            for vals in itertools.product(*[matrix[k] for k in cart_keys])] or [{}]
    for inc in matrix.get("include", []) or []:
        if not isinstance(inc, dict):
            continue
        overlap = [k for k in cart_keys if k in inc]
        if overlap:
            for combo in base:
                if all(combo.get(k) == inc[k] for k in overlap):
                    combo.update(inc)
        else:
            base.append(dict(inc))
    return base, cart_keys


def job_display_names(job_id: str, job: dict):
    """All possible status-check display names for one job (matrix-expanded)."""
    name = job.get("name")
    matrix = (job.get("strategy") or {}).get("matrix")
    if not matrix:
        return {str(name) if name else job_id}
    combos, cart_keys = matrix_combos(matrix)
    out = set()
    for combo in combos:
        if name and "${{" in str(name):
            disp = str(name)
            for k, v in combo.items():
                disp = disp.replace("${{ matrix.%s }}" % k, str(v))
                disp = disp.replace("${{matrix.%s}}" % k, str(v))
            out.add(disp)
        elif name:
            out.add(str(name))
        else:
            suffix = ", ".join(str(combo[k]) for k in cart_keys if k in combo)
            out.add(f"{job_id} ({suffix})")
    return out


def triggers_on_pr(doc: dict) -> bool:
    # YAML parses bare `on:` as the boolean key True.
    on = doc.get("on", doc.get(True))
    if on is None:
        return False
    if isinstance(on, str):
        return on == "pull_request"
    if isinstance(on, list):
        return "pull_request" in on
    if isinstance(on, dict):
        return "pull_request" in on
    return False


_EXIT_1_RE = re.compile(r"exit\s+1\b")


def _find_matching_close(text: str, open_ch: str, close_ch: str):
    """`text` must start with `open_ch`. Scans left-to-right tracking
    `open_ch`/`close_ch` depth, skipping over single/double-quoted
    substrings (so a `close_ch` inside a literal, e.g. a stray `"]"`, is
    never mistaken for a real closer). Returns the index of the `close_ch`
    that brings depth back to 0, or `None` if unbalanced. This is
    depth-based, not "starts-with/ends-with", so `[[ A ]]` and `[ A ]` are
    both handled by the SAME single-character depth counter without any
    special-casing of the double-bracket bash test syntax — two nested
    single-char opens/closes are structurally identical to one nested
    pair."""
    depth = 0
    in_quote = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if in_quote:
            if c == in_quote:
                in_quote = None
        elif c in ("'", '"'):
            in_quote = c
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _find_ghexpr_close(text: str):
    """`text` must start with `${{`. Same idea as `_find_matching_close`
    but for the 3-char/2-char `${{` / `}}` token pair (handles nested
    `${{ }}` and skips quoted substrings). Returns the index of the final
    `}` of the matching `}}`, or `None` if unbalanced."""
    depth = 0
    in_quote = None
    i, n = 0, len(text)
    while i < n:
        if in_quote:
            if text[i] == in_quote:
                in_quote = None
            i += 1
            continue
        if text[i] in ("'", '"'):
            in_quote = text[i]
            i += 1
            continue
        if text.startswith("${{", i):
            depth += 1
            i += 3
            continue
        if text.startswith("}}", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i - 1
            continue
        i += 1
    return None


def _strip_full_span_wrapper(text: str):
    """If `text` (already stripped) is spanned, in its ENTIRETY, by exactly
    one permitted syntactic wrapper — `${{ ... }}`, bash test brackets
    (`[ ... ]` or `[[ ... ]]`, handled uniformly since both are just one or
    two literal `[`/`]` characters — no special-casing needed), or a
    balanced `( ... )` — strip exactly that one outer layer and return the
    trimmed inner text. Returns `None` if none apply, INCLUDING the case
    where `text` merely *starts*/*ends* with the relevant character but the
    two are not a matching pair spanning the COMPLETE string (e.g. two
    adjacent bracket groups joined by `&&`: the first `[`'s matching `]` is
    not the last character of `text`, so this deliberately does not strip
    only the first group and silently drop the second).

    Quote characters are NEVER treated as a wrapper here — a whole
    comparison term like `"X" = "Y"` starts and ends with a quote char but
    is two independent quoted literals, not one wrapping pair. Operand-only
    quote-stripping is handled separately by `_normalize_operand`, which is
    only ever applied to an already-isolated lhs operand, never to a whole
    condition."""
    t = text.strip()
    if len(t) < 2:
        return None
    if t.startswith("${{") and t.endswith("}}"):
        close = _find_ghexpr_close(t)
        if close == len(t) - 1:
            return t[3:-2].strip()
        return None
    if t[0] == "[" and t[-1] == "]":
        close = _find_matching_close(t, "[", "]")
        if close == len(t) - 1:
            return t[1:-1].strip()
        return None
    if t[0] == "(" and t[-1] == ")":
        close = _find_matching_close(t, "(", ")")
        if close == len(t) - 1:
            return t[1:-1].strip()
        return None
    return None


# Boolean combinators that join two or more sibling terms in a bash `[ ]`/
# `[[ ]]` test or a GitHub Actions expression: `&&`, `||`, and the POSIX
# `test`/`[` binary operators `-a` (AND) / `-o` (OR). The `-a`/`-o` branches
# are word-bounded (no `\w` or `-` immediately before/after) so they never
# match inside an identifier like `gpu-export-closure` or `-action`.
_TOP_LEVEL_CONNECTOR_RE = re.compile(
    r'&&|\|\||(?<![\w-])-a(?![\w-])|(?<![\w-])-o(?![\w-])'
)


def _split_top_level_terms(text: str):
    """Split `text` on every occurrence of a boolean combinator (`&&`,
    `||`, `-a`, `-o`) that is NOT inside a single/double-quoted substring.
    Deliberately does NOT need bracket-depth tracking: a combinator that
    still appears here — whether between two sibling `[ ]` groups
    (`[ A ] && [ B ]`) or inside a single not-yet-unwrapped bracket
    (`[ A -a B ]`) — genuinely joins two or more sibling boolean terms
    either way, so both shapes correctly yield `len(terms) > 1` regardless
    of whether the enclosing bracket has already been peeled off by
    `_strip_full_span_wrapper` on this pass or a later one."""
    terms = []
    buf = []
    in_quote = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if in_quote:
            buf.append(c)
            if c == in_quote:
                in_quote = None
            i += 1
            continue
        if c in ("'", '"'):
            in_quote = c
            buf.append(c)
            i += 1
            continue
        m = _TOP_LEVEL_CONNECTOR_RE.match(text, i)
        if m:
            terms.append("".join(buf).strip())
            buf = []
            i = m.end()
            continue
        buf.append(c)
        i += 1
    terms.append("".join(buf).strip())
    return [t for t in terms if t]


_TRAILING_THEN_RE = re.compile(r';?\s*then\s*$')


def _reduce_to_single_active_term(cond_text: str):
    """The core round-14 structural fix. Repeatedly (a) strips a leading
    `if` keyword and a trailing `; then`/`then` (a no-op for job-level
    `if:` strings, which never start with `if `), (b) splits the remainder
    on top-level boolean combinators and REQUIRES exactly one term — if
    not, returns `None` immediately: this is the actual fix, applied at
    EVERY nesting depth as wrappers are peeled off, not just once, so an
    extra `&&`/`||`/`-a`/`-o` term cannot hide as a suffix, a prefix, a
    duplicate, or nested one level inside a `[[ ]]` that only a shallower
    pass would have unwrapped without re-splitting — and (c) strips one
    fully-spanning wrapper layer (`${{ }}`, `[ ]`/`[[ ]]`, `( )`) from that
    single term and loops, so stripping a bracket that exposes a
    previously-hidden `-a`/`-o` (`[ A -a B ]` -> `A -a B`) is re-validated
    on the next pass. Stabilizes when no wrapper strips further. Returns
    the final single, wrapper-free comparison candidate, or `None` if 0 or
    >=2 top-level terms were found at ANY depth."""
    text = cond_text.strip()
    if text == "if" or text.startswith("if ") or text.startswith("if\t"):
        text = text[2:]
    text = _TRAILING_THEN_RE.sub('', text.strip()).strip()

    for _ in range(32):  # defensive bound; text strictly shrinks each real iteration
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return None
        terms = _split_top_level_terms(text)
        if len(terms) != 1:
            return None
        text = terms[0]
        stripped = _strip_full_span_wrapper(text)
        if stripped is None:
            return text
        text = stripped
    return None  # unreachable in practice; fail closed if the bound is ever hit


def _normalize_operand(text: str) -> str:
    """Reduce an ALREADY-ISOLATED lhs operand (never a whole condition) to
    its bare path by repeatedly stripping one fully-spanning quote pair
    and/or one fully-spanning `${{ ... }}`/`( )` wrapper, e.g.
    `"${{ needs.gpu-export-closure.result }}"` ->
    `needs.gpu-export-closure.result`."""
    t = text.strip()
    for _ in range(8):
        if len(t) >= 2 and t[0] in ("'", '"') and t[-1] == t[0] and t[0] not in t[1:-1]:
            t = t[1:-1].strip()
            continue
        stripped = _strip_full_span_wrapper(t)
        if stripped is None:
            return t
        t = stripped
    return t


_SINGLE_COMPARISON_RE = re.compile(
    r'^(?P<lhs>.+?)\s*(?P<op>!=|==|=)\s*(?P<q>[\'"])(?P<lit>[^\'"]*)(?P=q)$'
)


def _equality_test(text: str, expr_substr: str, literal: str):
    """Return `True` if `text` is, structurally, a real bash-test or
    GitHub-Actions equality comparison of the form `<expr containing
    expr_substr> (=|==) "<literal>"` and NOTHING ELSE, `False` if the SAME
    comparison is an INEQUALITY test (`!=`) against that literal and
    NOTHING ELSE, or `None` if `expr_substr` cannot be found paired with
    `literal` via a recognized comparison operator that constitutes the
    ENTIRE condition (missing guard, an unrelated/typo'd variable
    substituted in its place, or — round-14 — an extra active boolean term
    combined with the comparison via `&&`/`||`/`-a`/`-o`).

    Round-13 fix-iteration 2 pinned the operator/direction and the exact
    variable path but only required the comparison to appear somewhere
    inside `text` (a bounded-window `re.search`), which an independent
    round-13 verifier proved accepts an always-false extra conjunct
    appended anywhere in the condition — e.g.
    `[ "${{ needs.gpu-export-closure.result }}" = "skipped" ] &&
    [ "1" = "0" ]` — because real bash's `&&` makes the WHOLE condition
    false whenever either operand is false, permanently dead-coding the
    guard body at runtime, while the window-search still finds the
    required comparison sitting right next to the anchor and reports the
    guard as present and correctly directed.

    Round-14 fix: `text` is first reduced, via `_reduce_to_single_active_term`,
    to a single top-level active boolean term — this REJECTS `text` outright
    (returns `None` here) if it contains a second top-level term joined by
    `&&`, `||`, `-a`, or `-o`, at ANY nesting depth (a suffix conjunct, a
    prefix conjunct, a duplicate/decoy comparison, or one nested a single
    level inside `[[ ]]`/`( )` are all rejected identically, since none of
    them change the fact that there is more than one top-level term). Only
    a single surviving term is then required to `fullmatch` (anchored at
    both ends, not `search`) the one recognized `<lhs> <op> "<literal>"`
    shape end to end, after which the lhs — itself unwrapped of its own
    local quoting/`${{ }}` via `_normalize_operand` — must equal
    `expr_substr` EXACTLY, not merely contain it.

    Determining "is there more than one top-level boolean term" is a purely
    syntactic question; this deliberately does NOT attempt to evaluate
    whether a second term is semantically always-true or always-false
    (undecidable in general for arbitrary shell/GH-expression text) —
    it forbids any second top-level term unconditionally, which is exactly
    what a validator that must never be defeated by a *specific* decoy
    literal requires.

    Handles both bash test syntax (`[ "${{ expr }}" = "literal" ]` /
    `!= "literal"`, `[[ ... ]]`, used inside `run:` shell blocks) and native
    GitHub Actions expression syntax (`expr != 'literal'`, used directly in
    a YAML `if:` field, unquoted and without `${{ }}` wrapping). Preserves
    every round-13 legitimate variant: `=`/`==`/`!=`, whitespace, and
    harmless outer parentheses."""
    term = _reduce_to_single_active_term(text)
    if term is None:
        return None
    m = _SINGLE_COMPARISON_RE.fullmatch(term)
    if not m:
        return None
    lhs = _normalize_operand(m.group("lhs"))
    if lhs != expr_substr or m.group("lit") != literal:
        return None
    return m.group("op") != "!="


def _strip_bash_comments(script: str) -> str:
    """Remove bash comments (a whole-line `#...` and a trailing ` #...`)
    from a shell script so a guard that exists only as commented-out dead
    code can never satisfy the structural checks below — round-13
    fix-iteration-1 closed exactly this bypass: commenting out only the
    live `if [ ... skipped ... ]; then ... fi` guard block while leaving an
    explanatory prose comment above it intact used to still make the old
    anchor-window scanner report the guard as present, because it never
    stripped comments before scanning.

    Line-oriented and does not understand shell quoting — sufficient for
    gate.yml's generated `run:` blocks, none of which embed a literal `#`
    inside a quoted string or a `${{ ... }}` expression (every step this
    validator reads was inspected for this)."""
    out_lines = []
    for line in script.splitlines():
        if line.lstrip().startswith("#"):
            continue
        hash_idx = line.find(" #")
        if hash_idx != -1:
            line = line[:hash_idx]
        out_lines.append(line)
    return "\n".join(out_lines)


def _is_block_open(line: str) -> bool:
    """True if `line`, stripped, OPENS a bash compound statement this
    validator's flat depth-counter tracks: an `if …; then` (the original,
    and by far the most common, shape in gate.yml), a `case … in` selector,
    or a `for …; do` / `while …; do` / `until …; do` loop header (any line
    ending in the bare word `do`, however it got there).

    Round 14, part 3 (this round's own further adversarial
    self-verification, after the `&&`-conjunct fix and the always-false
    if/fi-ancestor-wrapper fix above): a scanner that only recognizes
    `if`/`fi` is blind to a `case`/`for`/`while`/`until` wrapper around the
    guard — e.g. `case "$literally_never_this" in gpu-export-closure) <the
    whole real, correctly-worded guard> ;; esac` — real bash never enters
    that case arm (the selector never matches the pattern), but an if/fi-only
    tracker doesn't know `case`/`esac` bound anything, so it would still see
    the nested `if [ ... skipped ... ]; then …` as sitting at the TOP level
    of the script and report the guard present. Recognizing these additional
    openers (and `_is_block_close`'s matching `esac`/`done`) closes the
    identical "guard looks present, dead at runtime" outcome spelled with a
    different bash keyword. gate.yml's own real `for entry in … ; do … done`
    blocking loop (a genuine, unwrapped, top-level sibling of both guards) is
    unaffected: it simply becomes its own top-level entry, which never
    matches either `_equality_test()` anchor, exactly like the case before
    this line existed."""
    s = line.strip()
    if (s == "if" or s.startswith("if ") or s.startswith("if\t")) and s.endswith("then"):
        return True
    if (s.startswith("case ") or s.startswith("case\t")) and (
        s == "case in" or s.endswith(" in") or s.endswith("\tin")
    ):
        return True
    if s == "do" or s.endswith(" do") or s.endswith("\tdo"):
        return True
    return False


def _is_block_close(line: str) -> bool:
    """True if `line`, stripped, is exactly one of the closing keywords
    (`fi`, `esac`, `done`) that pair with `_is_block_open`'s openers. A flat
    depth counter that increments on ANY recognized opener and decrements on
    ANY recognized closer — without tracking which specific opener a given
    closer belongs to — is sound here: well-formed bash is properly
    lexically nested (a `case` can never close with `fi`, an `if` can never
    close with `esac`, in any script that actually parses), and this
    validator only ever runs on gate.yml's own real, already-valid `run:`
    steps or hand-built synthetic bash shaped the same way."""
    return line.strip() in ("fi", "esac", "done")


def _iter_if_blocks(script: str):
    """Yield `(condition_line, body_text)` for every top-level compound
    statement (`if …; then` / `fi`, `case … in` / `esac`, or
    `for`/`while`/`until … ; do` / `done` — see `_is_block_open`) found in
    `script`. Nested blocks are left verbatim inside a parent's
    `body_text` — call this again on a `body_text` to descend one level,
    which is exactly how `_has_gpu_export_closure_skip_guard` below ties a
    nested `docs_only` check to the SAME outer block that tested
    `needs.gpu-export-closure.result`, rather than merely finding both
    strings anywhere in the file within some fixed character distance.

    Stack-based on the recognized opener/closer keywords appearing as the
    entire remaining content of a line (after stripping) or as the leading
    (`_is_block_open`) token of a condition/selector/loop-header — this is
    the exact style every `run:` step in gate.yml uses (a condition always
    closes with `; then`/`; do` at end-of-line; a matching closer always
    sits alone on its own line). Not a general POSIX-sh parser — sufficient
    for the only input this function is ever fed: gate.yml-shaped bash,
    real or synthetic. Only `if`-shaped entries can ever satisfy
    `_equality_test()`'s comparison shape downstream; `case`/`for`/`while`/
    `until` entries are still yielded like any other top-level block (so a
    guard hidden inside one is correctly seen as NOT top-level and NOT
    matched — see `_is_block_open`), they just never themselves match the
    required `<lhs> <op> "<literal>"` comparison."""
    lines = script.splitlines()
    n = len(lines)
    blocks = []
    i = 0
    while i < n:
        if _is_block_open(lines[i]):
            cond_line = lines[i]
            depth = 1
            body: list[str] = []
            j = i + 1
            while j < n and depth > 0:
                if _is_block_open(lines[j]):
                    depth += 1
                    body.append(lines[j])
                elif _is_block_close(lines[j]):
                    depth -= 1
                    if depth == 0:
                        break
                    body.append(lines[j])
                else:
                    body.append(lines[j])
                j += 1
            blocks.append((cond_line, "\n".join(body)))
            i = j + 1
            continue
        i += 1
    return blocks


def _strip_nested_if_blocks(text: str) -> str:
    """Remove every nested compound-statement span (an `_is_block_open`
    opener through its matching `_is_block_close` closer, inclusive, at
    whatever depth it starts — `if …/fi`, `case …/esac`, or
    `for`/`while`/`until …/done`) from `text`, leaving only the lines that
    sit OUTSIDE of any such block — the part of `text` bash executes
    unconditionally, with no further guard in between. Used to verify a
    required token (`exit 1`) is present UNCONDITIONALLY in a matched
    block's own body, not hidden one level deeper inside a further wrapper
    whose own condition (or, for a loop, whose own iteration list, or, for
    a case, whose own selector) this validator does not — and in general
    cannot — prove is always taken.

    Round 14, part 2 (this round's own adversarial self-verification, after
    the `&&`-conjunct fix above): a raw substring/regex search over `text`
    cannot tell `exit 1` sitting bare in a block's body apart from `exit 1`
    sitting inside `if [ "1" = "0" ]; then exit 1; fi` nested inside that
    body — the latter is real, permanent dead code in bash (the wrapper
    condition never being true), but is textually indistinguishable from
    the former to any search that does not know where block boundaries are.
    Part 3 widened the recognized openers/closers beyond `if`/`fi` to
    `case`/`esac` and `for`/`while`/`until`/`do`/`done` for the identical
    reason `_is_block_open` was widened — `exit 1` hidden inside
    `case "$literally_never" in x) exit 1 ;; esac`, or inside a
    zero-iteration `for x in; do exit 1; done`, is just as dead as inside a
    `[ "1" = "0" ]` if-wrapper, and just as invisible to an if/fi-only
    stripper."""
    lines = text.splitlines()
    n = len(lines)
    out: list[str] = []
    i = 0
    while i < n:
        if _is_block_open(lines[i]):
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if _is_block_open(lines[j]):
                    depth += 1
                elif _is_block_close(lines[j]):
                    depth -= 1
                j += 1
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _has_gpu_export_closure_skip_guard(script: str) -> bool:
    """True iff `script` contains a TOP-LEVEL if-block whose OWN condition
    line tests `needs.gpu-export-closure.result` for EQUALITY (`=`/`==`,
    not `!=`) against the literal `skipped` (the variable and the
    operator/literal must be directly adjacent in that SAME condition
    line — not just somewhere earlier in the file, and not merely both
    substrings present anywhere in the line), AND — as a further TOP-LEVEL
    if-block nested DIRECTLY inside that same block's own body (i.e. not
    itself wrapped inside yet another if-block) — a further if-block whose
    own condition tests the EXACT path `needs.detect-impact.outputs.
    docs_only` for INEQUALITY (`!=`, not `=`/`==`) against the literal
    `true`, and whose own body contains a literal `exit 1` that itself
    sits OUTSIDE of any further nested if-block (`_strip_nested_if_blocks`).

    Tying all anchors to one conditional hierarchy (rather than a fixed
    character-distance window over the joined script text) means:
    - a correctly-scoped, real guard for a DIFFERENT job (e.g.
      caas-security's own docs_only-gated pre-check) can never satisfy
      gpu-export-closure's own requirement, because its condition line
      does not mention `needs.gpu-export-closure.result`;
    - a bare, unguarded mention of `needs.gpu-export-closure.result` (e.g.
      the step-summary echo line, or the generic loop's
      `"gpu-export-closure:${{ needs.gpu-export-closure.result }}:may_skip"`
      entry) never counts, because it never opens an if-block at all;
    - `docs_only`/`exit 1` appearing anywhere else in the script (e.g. in
      the unrelated failure/cancelled loop) cannot be borrowed to complete
      gpu-export-closure's guard, because they must be inside the body of
      the specific if-block keyed on gpu-export-closure's own result;
    - round-13 fix-iteration 2: an INVERTED comparison in either
      condition (`!=` instead of `=`/`==` on the outer `skipped` test, or
      `=`/`==` instead of `!=` on the inner `docs_only` test) is now
      rejected even though every substring the old presence-only check
      looked for is still present. Likewise, an unrelated or typo'd
      variable standing in for the real
      `needs.detect-impact.outputs.docs_only` path (e.g.
      `env.SOME_UNRELATED_docs_only_LOOKALIKE`) no longer satisfies the
      inner check, because `_equality_test()` requires the EXACT context
      path adjacent to the operator/literal, not merely the substring
      `docs_only` anywhere in the condition;
    - round 14, part 2 (this round's own adversarial self-verification):
      an earlier version of this function searched for the outer and
      inner conditions at ANY nesting depth via a recursive
      `_walk_all_if_blocks()` helper, reasoning that "the guard might
      legitimately sit under some other wrapper". That recursion is
      exactly what let a whole guard — or just its inner docs_only
      pre-check — be wrapped inside an unrelated, always-false ancestor
      if-block (e.g. `if [ "1" = "0" ]; then <the entire real guard>
      fi`): the recursive search still descended into that ancestor's
      body and found the real, correctly-worded, correctly-directed
      comparison sitting inside it, and reported the guard present, even
      though real bash never enters the ancestor's body at all — a
      compound dead-guard shape achieved via nesting instead of a `&&`
      conjunct in a single condition line. The real, unmutated gate.yml
      never nests either guard inside anything else (both the outer
      `skipped` test and the inner `docs_only` test sit at the top level
      of their respective scopes), so requiring TOP-LEVEL-ONLY matches
      (`_iter_if_blocks`, not `_walk_all_if_blocks`) for both the outer
      script and the inner body loses no legitimate coverage while
      closing this. The matched inner block's own body is additionally
      run through `_strip_nested_if_blocks()` before the `exit 1` search,
      so the identical trick cannot be applied one level deeper still —
      wrapping just the `exit 1` itself in a further always-false
      if-block inside an otherwise-correct, unwrapped outer/inner pair."""
    for cond_line, body in _iter_if_blocks(script):
        outer = _equality_test(cond_line, "needs.gpu-export-closure.result", "skipped")
        if outer is not True:
            continue
        for inner_cond, inner_body in _iter_if_blocks(body):
            inner = _equality_test(
                inner_cond, "needs.detect-impact.outputs.docs_only", "true"
            )
            if inner is not False:
                continue
            if _EXIT_1_RE.search(_strip_nested_if_blocks(inner_body)):
                return True
    return False


def _render_gh_expr_tokens(script: str, substitutions: dict) -> "str | None":
    """Replace every `${{ <key> }}` token in `script` with
    `substitutions[key]` when present, otherwise the benign literal
    `"success"` (syntactically safe everywhere gate.yml's `final-verdict`
    script ever uses a token — always a plain string-comparison operand,
    never inside an unquoted arithmetic or command-substitution context).
    Returns `None` if any `${{ ... }}` token remains after the pass —
    should be unreachable given the unconditional fallback, but a caller
    must never execute text that still contains a raw GitHub Actions
    expression as if it were plain bash."""
    def _sub(m: "re.Match") -> str:
        key = m.group(1).strip()
        return substitutions.get(key, "success")
    rendered = re.sub(r"\$\{\{\s*([^}]+?)\s*\}\}", _sub, script)
    if re.search(r"\$\{\{.*?\}\}", rendered):
        return None
    return rendered


def _gpu_export_closure_guard_blocks_at_runtime(script: str) -> "bool | None":
    """Behavioral defense-in-depth (round 14, part 4). Renders `script` (the
    comment-stripped `final-verdict` `run:` text) for the exact issue #335
    round-12 danger scenario — `needs.gpu-export-closure.result` = `skipped`,
    `needs.detect-impact.outputs.docs_only` = `false`, every other `${{ }}`
    token a benign `success` — and executes the result via a real `bash -c`
    subprocess. Returns `True` if the rendered script exits non-zero (blocks,
    as the round-13/14 contract requires), `False` if it exits `0` (the
    false-green), or `None` if the script could not be rendered/executed at
    all (an unsubstituted token, no `bash` on `PATH`, or a timeout) —
    callers MUST treat `None` as "could not verify", never as a pass.

    No finite set of static syntactic rules can prove an arbitrary shell
    fragment's `exit 1` is UNCONDITIONALLY reached — bash is Turing-complete.
    `_reduce_to_single_active_term()` closes the `&&`/`||`/`-a`/`-o` conjunct
    class on the single recognized comparison; widening `_iter_if_blocks()`
    / `_strip_nested_if_blocks()` to `if`/`case`/`for`/`while`/`until`
    (parts 2-3 above) closes every always-dead BLOCK-shaped wrapper. This
    round's own further self-adversarial testing found a THIRD shape,
    orthogonal to both: a short-circuited `<test> && exit 1` /
    `<test> || exit 1` STATEMENT — no `if`/`case`/`for`/`while`/`until`
    keyword at all, so there is no block for `_is_block_open()` to
    recognize or for `_strip_nested_if_blocks()` to strip, yet
    `[ "1" = "0" ] && exit 1` never runs `exit 1` and `[ "1" = "1" ] ||
    exit 1` never runs it either — both are just as permanently dead as
    any of the block-shaped tricks already closed. Rather than add a
    fourth, fifth, ... special case per newly-imagined shell construct,
    this function proves the actual, end-to-end runtime behavior for the
    one scenario that matters directly — the same technique
    `check_final_verdict_gpu_export_closure_skip_policy_fixtures` already
    uses against the real, live gate.yml as a test fixture, now also
    applied INSIDE the gate function itself so it protects every caller
    (the CI gate's own `main()` AND every synthetic fixture that calls
    `check_gpu_export_closure_skip_guard()` directly), not just the one
    real file a separate test happens to read. A script that passes every
    structural check above and still fails this behavioral one is a
    compound-dead-guard false green by another name, regardless of which
    bash construct — known or not yet imagined — achieves it."""
    substitutions = {
        "needs.gpu-export-closure.result": "skipped",
        "needs.detect-impact.outputs.docs_only": "false",
    }
    rendered = _render_gh_expr_tokens(script, substitutions)
    if rendered is None:
        return None
    try:
        proc = subprocess.run(
            ["bash", "-c", rendered], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.returncode != 0


def check_gpu_export_closure_skip_guard(gate_doc: dict) -> list[str]:
    """Structural validator for the issue #335 round-13 contract. Returns a
    list of violation strings (empty = compliant). Pure function over an
    ALREADY-PARSED gate.yml YAML dict (e.g. from yaml.safe_load) — no file
    I/O of its own, so it can be exercised directly against synthetic
    fixtures without a real workflow file on disk.

    Round-13 fix-iteration 2: every docs_only-related check below (the
    job-level `if:` guard and the nested pre-check inside final-verdict's
    script) validates via `_equality_test()`, not bare substring presence.
    An independent verifier proved the fix-iteration-1 version — which
    required the right tokens to co-occur inside the right nested if-block,
    but never checked the comparison OPERATOR/DIRECTION or that the
    referenced variable was the REAL `needs.detect-impact.outputs.docs_only`
    path — still reported zero violations for a script with `!=` flipped to
    `=`/`==` (or vice versa) in either comparison, or with an
    unrelated/typo'd variable substituted for the real docs_only path. Each
    such script would NOT actually fail closed for a non-docs skip at
    runtime, despite looking structurally compliant to presence-only
    checks."""
    violations: list[str] = []
    jobs = gate_doc.get("jobs") or {}

    if "gpu-export-closure" not in jobs:
        violations.append("gpu-export-closure job is missing from gate.yml")

    if "final-verdict" not in jobs:
        violations.append("final-verdict job is missing from gate.yml")
        return violations

    final_verdict = jobs.get("final-verdict") or {}

    needs = final_verdict.get("needs") or []
    if isinstance(needs, str):
        needs = [needs]
    if "gpu-export-closure" not in needs:
        violations.append(
            "final-verdict does not depend on gpu-export-closure via needs:"
        )

    if "gpu-export-closure" in jobs:
        gpu_job = jobs.get("gpu-export-closure") or {}
        if_guard = str(gpu_job.get("if", ""))
        # Round-13 fix-iteration 2: check the comparison's DIRECTION and the
        # EXACT context path, not merely substring presence of "docs_only"
        # anywhere in the guard — an independent verifier proved a guard
        # inverted to `== 'true'` (which runs the job ONLY on docs-only
        # changes, the exact opposite of the required "must run for every
        # non-docs change" policy) passed the old presence-only check.
        guard_dir = _equality_test(
            if_guard, "needs.detect-impact.outputs.docs_only", "true"
        )
        if guard_dir is None:
            violations.append(
                "gpu-export-closure's own if: guard is not scoped to the exact "
                "needs.detect-impact.outputs.docs_only path compared to 'true' "
                "via a recognized operator — it would run/skip unconditionally "
                "regardless of whether the change is docs-only, or is wired to "
                "an unrelated/typo'd variable"
            )
        elif guard_dir is not False:
            violations.append(
                "gpu-export-closure's own if: guard tests docs_only for "
                "EQUALITY to 'true' (=/==) instead of INEQUALITY (!=) — this "
                "is an inverted guard that would run the job ONLY on "
                "docs-only changes and skip it for every real code change, "
                "the exact opposite of the required policy"
            )

    steps = final_verdict.get("steps") or []
    script_parts = [
        str(step.get("run"))
        for step in steps
        if isinstance(step, dict) and step.get("run") is not None
    ]
    if not script_parts:
        violations.append(
            "final-verdict has no run: step to evaluate — cannot verify the "
            "skip-guard pre-check exists"
        )
        return violations

    # Strip bash comments BEFORE any structural scan: a guard that exists
    # only as commented-out dead code must never count (see
    # _strip_bash_comments' docstring for the exact bypass this closes).
    script = _strip_bash_comments("\n".join(script_parts))

    if not _has_gpu_export_closure_skip_guard(script):
        violations.append(
            "final-verdict's script has no docs_only-gated pre-check nested in "
            "the SAME if-block as a check of needs.gpu-export-closure.result == "
            "'skipped' — this is exactly the round-12 false-green shape (an "
            "unconditional non-blocking skip policy with no scoped guard), or a "
            "guard that only exists for a different job, only as a comment, only "
            "as an unguarded mention (e.g. the step-summary echo line), has the "
            "comparison direction inverted (!= instead of =/== on the 'skipped' "
            "test, or =/== instead of != on the docs_only test), or tests an "
            "unrelated/typo'd variable instead of the exact "
            "needs.detect-impact.outputs.docs_only path"
        )
    else:
        # Round 14, part 4: a structurally-recognized guard can still be
        # runtime-dead via a bash construct no static check enumerates (a
        # short-circuited `&&`/`||` exit instead of a block keyword — see
        # _gpu_export_closure_guard_blocks_at_runtime's docstring). Prove it
        # actually blocks for the real round-12 danger scenario before
        # trusting the structural result — a behavioral assertion, not
        # merely a structural one, per acceptance criterion 5.
        runtime_blocks = _gpu_export_closure_guard_blocks_at_runtime(script)
        if runtime_blocks is not True:
            violations.append(
                "final-verdict's script structurally APPEARS to have a "
                "docs_only-gated pre-check for gpu-export-closure (a "
                "top-level if-block testing needs.gpu-export-closure.result "
                "== 'skipped' with a nested docs_only != 'true' check and an "
                "exit 1), but a real `bash -c` execution of the script under "
                "the round-12 danger scenario (gpu-export-closure skipped on "
                "a non-docs change) did not exit non-zero "
                + ("(execution could not be completed/verified)"
                   if runtime_blocks is None
                   else "(exited 0)")
                + " — the guard is real dead code at runtime despite passing "
                "every static structural check, e.g. a short-circuited "
                "`&&`/`||` exit statement instead of an if/case/for/while/"
                "until block"
            )

    # Pre-existing invariant that must not regress: failure/cancelled results
    # stay unconditionally blocking. Looser on purpose — the goal is to catch
    # deletion of the whole loop, not to over-specify its exact shell syntax.
    # Comments were already stripped above, so a commented-out loop cannot
    # satisfy this either.
    blocking_found = False
    for fail_m in re.finditer(r"failure", script):
        window = script[fail_m.start():fail_m.start() + 400]
        if "cancelled" in window and _EXIT_1_RE.search(window):
            blocking_found = True
            break
    if not blocking_found:
        violations.append(
            "final-verdict's script no longer appears to treat failure/cancelled "
            "results as unconditionally blocking"
        )

    return violations


def main() -> int:
    if not REQUIRED_SH.exists():
        print(f"check_required_checks_match_jobs: {REQUIRED_SH} not found — run from "
              "the UltrafastSecp256k1 submodule root")
        return 1

    contexts = parse_contexts(REQUIRED_SH.read_text())
    if contexts is None:
        fail(f"could not parse a contexts array from {REQUIRED_SH}")
        return 1

    # name -> set of workflow files that emit it on a PR-triggered workflow
    pr_names = {}
    all_names = {}
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        try:
            doc = yaml.safe_load(wf.read_text())
        except yaml.YAMLError as e:
            print(f"  (warn) could not parse {wf.name}: {e}")
            continue
        if not isinstance(doc, dict):
            continue
        on_pr = triggers_on_pr(doc)
        for job_id, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for disp in job_display_names(job_id, job):
                all_names.setdefault(disp, set()).add(wf.name)
                if on_pr:
                    pr_names.setdefault(disp, set()).add(wf.name)

    errors = 0
    for ctx in contexts:
        if ctx in pr_names:
            print(f"  [ok] {ctx}")
        elif ctx in all_names:
            fail(f"context '{ctx}' matches a job in {sorted(all_names[ctx])} but that "
                 "workflow does NOT trigger on pull_request — strict mode would block "
                 "every merge forever")
            errors += 1
        else:
            fail(f"context '{ctx}' matches NO job display name in any workflow "
                 "(stale/phantom required check)")
            errors += 1

    context_errors = errors

    # Second, independent check: the issue #335 round-13 gpu-export-closure
    # skip-guard contract, validated structurally against the parsed gate.yml
    # (not the required-contexts list).
    gpu_guard_violations: list = []
    gate_yml = WORKFLOWS / "gate.yml"
    if not gate_yml.exists():
        print(f"  (warn) {gate_yml} not found — skipping gpu-export-closure "
              "skip-guard check")
    else:
        try:
            gate_doc = yaml.safe_load(gate_yml.read_text())
        except yaml.YAMLError as e:
            print(f"  (warn) could not parse {gate_yml.name}: {e}")
            gate_doc = None
        if isinstance(gate_doc, dict):
            gpu_guard_violations = check_gpu_export_closure_skip_guard(gate_doc)
            for violation in gpu_guard_violations:
                fail(violation)
        elif gate_doc is not None:
            fail(f"{gate_yml} did not parse to a mapping — cannot validate the "
                 "gpu-export-closure skip-guard contract")
            gpu_guard_violations = ["unparseable gate.yml"]

    errors += len(gpu_guard_violations)

    if errors:
        print(f"check_required_checks_match_jobs: {context_errors} unresolved "
              f"context(s), {len(gpu_guard_violations)} gpu-export-closure "
              "skip-guard violation(s)")
        return 1
    print(f"check_required_checks_match_jobs: all {len(contexts)} required contexts "
          "resolve to PR-triggered jobs, and the gpu-export-closure skip-guard "
          "contract (issue #335 round 13) is structurally intact [PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
