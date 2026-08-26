// ============================================================================
// test_shim_security_gate_policy.cpp -- Source-coupled contract regression for
// the "Run shim security regression modules" step in .github/workflows/gate.yml
// ============================================================================
//
// This is NOT a runtime behavioral test of unified_audit_runner -- it is a
// contract test against the CI YAML source text itself, in two layers:
//
//   1. Static substring/ordering checks against the raw YAML text (bounded,
//      not a YAML parser) confirming every documented property is present.
//   2. Executable scenario checks: the step's `run:` script is extracted from
//      the YAML text and dedented (mirroring plain YAML block-scalar
//      indentation stripping -- not a general YAML parser), the external
//      `unified_audit_runner` invocation is swapped for a controlled stub,
//      and the REAL extracted bash+python pipeline is executed end-to-end
//      via bash/python3 against synthetic audit reports. This exercises the
//      actual control flow (not just token presence), which is what layer 1
//      cannot catch on its own -- see CI_SHIM_GATE_REPORT_POLICY_ARTIFACT_703,
//      where a static-only version of this test passed against a patch whose
//      runner-exit-code reconciliation unconditionally hard-failed the exact
//      advisory-only case its own policy said must not hard-fail.
//
// Layer 2 requires `bash` and `python3` on PATH; if either is unavailable
// (e.g. a Windows/MSVC-only environment) the executable scenarios are
// skipped with a notice and only layer 1 runs -- this file is not yet wired
// into unified_audit_runner's advisory/non-advisory classification, so no
// ADVISORY_SKIP_CODE convention applies here (see file-level note at the
// bottom of this file about deferred wiring).
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  define popen  _popen
#  define pclose _pclose
#else
#  include <sys/wait.h>
#endif

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, (msg)); } \
} while(0)

namespace {

const char* kGateYmlPath = ".github/workflows/gate.yml";
const char* kStepStartMarker = "- name: Run shim security regression modules";
const char* kNextStepMarker = "\n      - name:";
const char* kRunnerInvocation =
    "./out/ci-shim/audit/unified_audit_runner --json-only --report-dir .";

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// True if every needle in `needles` appears in `haystack`, each strictly
// after the position where the previous one was found -- i.e. the needles
// occur in the given order (not necessarily contiguous).
bool contains_in_order(const std::string& haystack,
                        std::initializer_list<const char*> needles) {
    size_t pos = 0;
    for (const char* needle : needles) {
        size_t found = haystack.find(needle, pos);
        if (found == std::string::npos) return false;
        pos = found + std::string(needle).size();
    }
    return true;
}

// Dedent a YAML block-scalar body: strip the leading-whitespace prefix of
// the first non-blank line from every line, mirroring what a YAML parser
// does to a `run: |` scalar before bash ever sees the text. Bounded to this
// one specific, already-known convention -- not a general YAML parser.
std::string dedent_block_scalar(const std::string& raw) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (true) {
        size_t nl = raw.find('\n', start);
        if (nl == std::string::npos) { lines.push_back(raw.substr(start)); break; }
        lines.push_back(raw.substr(start, nl - start));
        start = nl + 1;
    }
    size_t base_indent = std::string::npos;
    for (const auto& line : lines) {
        size_t first_non_space = line.find_first_not_of(' ');
        if (first_non_space == std::string::npos) continue;
        base_indent = first_non_space;
        break;
    }
    if (base_indent == std::string::npos || base_indent == 0) return raw;
    std::string out;
    for (size_t li = 0; li < lines.size(); ++li) {
        const std::string& line = lines[li];
        size_t first_non_space = line.find_first_not_of(' ');
        size_t strip = (first_non_space == std::string::npos)
            ? line.size()
            : std::min(base_indent, first_non_space);
        out += line.substr(strip);
        if (li + 1 < lines.size()) out += '\n';
    }
    return out;
}

bool extract_run_script(const std::string& block, std::string& out_script) {
    const std::string marker = "run: |\n";
    size_t pos = block.find(marker);
    if (pos == std::string::npos) return false;
    out_script = dedent_block_scalar(block.substr(pos + marker.size()));
    return true;
}

// Runs `cmd` via a shell, capturing combined stdout+stderr (caller is
// expected to append `2>&1` to `cmd` when stderr matters) and returning the
// child's real exit code (not the raw wait-status pclose() gives on POSIX).
int run_capture(const std::string& cmd, std::string& out) {
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return -1;
    char buf[4096];
    out.clear();
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0) out.append(buf, n);
    int status = pclose(fp);
    if (status == -1) return -1;
#ifdef _WIN32
    return status;
#else
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

bool tool_available(const char* probe_cmd) {
    std::string out;
    return run_capture(probe_cmd, out) == 0;
}

struct Scenario {
    const char* name;
    std::string stub;
    int expected_exit;
    const char* must_contain;
    const char* must_not_contain; // nullptr if none
};

std::string stub_write_report(const std::string& json, int rc) {
    return "(printf '%s' '" + json + "' > audit_report.json; exit " + std::to_string(rc) + ")";
}

bool run_scenario(const std::string& dedented_script,
                   const std::filesystem::path& scratch_root,
                   const Scenario& sc) {
    size_t pos = dedented_script.find(kRunnerInvocation);
    if (pos == std::string::npos) {
        printf("  FAIL [scenario:%s] runner invocation line not found in extracted script\n", sc.name);
        return false;
    }
    std::string script = dedented_script;
    script.replace(pos, std::string(kRunnerInvocation).size(), sc.stub);

    namespace fs = std::filesystem;
    fs::path dir = scratch_root / sc.name;
    std::error_code ec;
    fs::create_directories(dir, ec);

    fs::path script_path = dir / "run_step.sh";
    { std::ofstream f(script_path, std::ios::binary); f << script; }

    std::string cmd = "cd \"" + dir.string() + "\" && bash run_step.sh 2>&1";
    std::string out;
    int rc = run_capture(cmd, out);

    fs::remove_all(dir, ec);

    bool ok = (rc == sc.expected_exit)
        && (out.find(sc.must_contain) != std::string::npos)
        && (sc.must_not_contain == nullptr || out.find(sc.must_not_contain) == std::string::npos);
    if (!ok) {
        printf("  FAIL [scenario:%s] exit=%d (expected %d) output=%s\n",
               sc.name, rc, sc.expected_exit, out.c_str());
    }
    return ok;
}

} // namespace

int test_shim_security_gate_policy_run() {
    printf("[shim-gate-report-policy] Contract checks against %s\n", kGateYmlPath);
    g_pass = g_fail = 0;

    std::string gate_yml;
    if (!read_file(kGateYmlPath, gate_yml)) {
        printf("  FAIL [%s:%d] could not read %s (wrong cwd? test must run from repo root)\n",
               __FILE__, __LINE__, kGateYmlPath);
        return 1;
    }

    size_t step_start = gate_yml.find(kStepStartMarker);
    CHECK(step_start != std::string::npos, "gate.yml: shim security regression step marker found");
    if (step_start == std::string::npos) {
        printf("  [shim-gate-report-policy] %d passed, %d failed\n", g_pass, g_fail);
        return g_fail;
    }

    size_t step_end = gate_yml.find(kNextStepMarker, step_start + std::string(kStepStartMarker).size());
    CHECK(step_end != std::string::npos, "gate.yml: next step marker found (bounds the step block)");
    std::string block = (step_end == std::string::npos)
        ? gate_yml.substr(step_start)
        : gate_yml.substr(step_start, step_end - step_start);

    // 1/2. errexit must not be able to swallow the report: disable it around
    // the runner invocation, capture the exit code, re-enable it, and check
    // the report file exists BEFORE renaming/parsing it.
    CHECK(contains_in_order(block, {"set +e", "unified_audit_runner", "RUNNER_RC=$?", "set -e"}),
          "runner invocation is bracketed by set +e / capture rc / set -e, in order");
    CHECK(contains_in_order(block, {"RUNNER_RC=$?", "audit_report.json", "mv audit_report.json shim_audit_report.json"}),
          "report existence is checked before the rename, after capturing the exit code");
    CHECK(block.find("mv audit_report.json shim_audit_report.json") != std::string::npos,
          "successful runs still rename audit_report.json -> shim_audit_report.json");

    // 3. GPU-advisory false-pass check (Rule 16 / CI-012) must survive.
    CHECK(block.find("advisory_false_pass") != std::string::npos,
          "GPU-advisory false-pass (Rule 16 / CI-012) classification is present");
    CHECK(block.find("ADVISORY_SKIP_CODE (77), not 0.") != std::string::npos,
          "Rule 16 violation message is present");

    // 4. Non-advisory hard-fail check must survive.
    CHECK(contains_in_order(block, {"failed = [m.get(", "not m.get(\"advisory\")"}),
          "non-advisory module failures are collected");
    CHECK(block.find("Shim security modules FAILED") != std::string::npos,
          "non-advisory module failures hard-fail the gate");

    // 5. Advisory-only-failure policy must be explicit, not silently dropped.
    CHECK(block.find("advisory_failed") != std::string::npos,
          "advisory-only failures are classified separately (advisory_failed)");
    CHECK(block.find("not in (0, 77)") != std::string::npos,
          "advisory-only failures are distinguished from skip(77)/false-pass(0)");

    // 6. The runner's own exit code must still be authoritative once errexit
    // no longer enforces it for us, AND (CI_SHIM_GATE_REPORT_POLICY_ARTIFACT_703
    // rework) that check must NOT be unconditional -- it must be reconciled
    // against advisory_failed so a non-zero exit fully explained by an
    // advisory-only failure passes (with a warning) instead of hard-failing,
    // while an unexplained non-zero exit still hard-fails. A prior version of
    // this step had an unconditional `if runner_rc != 0: sys.exit(1)` here,
    // which contradicted its own advisory-only-failure policy; this ordering
    // check fails against that exact regression.
    CHECK(contains_in_order(block, {"runner_rc = int(sys.argv[1])", "if runner_rc != 0:"}),
          "runner exit code is parsed and checked after JSON classification");
    CHECK(contains_in_order(block, {"if runner_rc != 0:", "if advisory_failed:",
                                     "not hard-failing per documented advisory-only policy",
                                     "else:", "unexplained hard failure", "sys.exit(1)"}),
          "a non-zero runner exit is reconciled against advisory_failed, not unconditionally hard-failed");

    // 7. No blanket-success escape hatches around the classification logic.
    CHECK(block.find("|| true") == std::string::npos,
          "no '|| true' blanket-success escape hatch in the step");
    CHECK(block.find("continue-on-error") == std::string::npos,
          "no continue-on-error escape hatch in the step");

    // Upload step: the artifact must be required, not merely warned-about,
    // now that the report is guaranteed to exist whenever this step doesn't
    // already hard-fail the job.
    size_t upload_start = gate_yml.find("- name: Upload shim audit report");
    CHECK(upload_start != std::string::npos, "gate.yml: upload shim audit report step found");
    if (upload_start != std::string::npos) {
        size_t upload_end = gate_yml.find(kNextStepMarker, upload_start + 10);
        std::string upload_block = (upload_end == std::string::npos)
            ? gate_yml.substr(upload_start)
            : gate_yml.substr(upload_start, upload_end - upload_start);
        CHECK(upload_block.find("if-no-files-found: error") != std::string::npos,
              "shim audit report upload fails loudly (not warn) if the artifact is missing");
    }

    // Executable/mutation scenarios (CI_SHIM_GATE_REPORT_POLICY_ARTIFACT_703
    // rework): extract the real run script, dedent it exactly as YAML would,
    // swap the external runner for a controlled stub, and execute the actual
    // bash+python pipeline end-to-end for each policy outcome.
    bool have_bash = tool_available("bash --version");
    bool have_python3 = tool_available("python3 --version");
    if (!have_bash || !have_python3) {
        printf("  [shim-gate-report-policy] bash/python3 unavailable in this environment -- "
               "skipping executable scenario checks (static contract checks above still apply)\n");
    } else {
        std::string dedented;
        if (!extract_run_script(block, dedented)) {
            CHECK(false, "gate.yml: could not extract+dedent the step's run script for executable checks");
        } else {
            namespace fs = std::filesystem;
            fs::path scratch_root = fs::temp_directory_path() / "ufsecp_shim_gate_policy_test";
            std::error_code ec;
            fs::create_directories(scratch_root, ec);

            const std::string json_advisory_only =
                R"({"sections": [{"modules": [{"id": "ok1", "advisory": false, "passed": true, "return_code": 0}, {"id": "test_exploit_gpu_scan", "advisory": true, "passed": false, "return_code": 99}]}]})";
            const std::string json_mixed =
                R"({"sections": [{"modules": [{"id": "core_regression", "advisory": false, "passed": false, "return_code": 1}, {"id": "test_exploit_gpu_scan", "advisory": true, "passed": false, "return_code": 99}]}]})";
            const std::string json_all_pass =
                R"({"sections": [{"modules": [{"id": "ok1", "advisory": false, "passed": true, "return_code": 0}, {"id": "ok2", "advisory": true, "passed": true, "return_code": 77}]}]})";
            const std::string json_gpu_false_pass =
                R"({"sections": [{"modules": [{"id": "test_exploit_gpu_scan", "advisory": true, "passed": true, "return_code": 0}]}]})";

            std::vector<Scenario> scenarios = {
                {"missing-report", "false", 1,
                 "cannot classify results, treating as a hard failure", nullptr},
                {"advisory-only-nonzero", stub_write_report(json_advisory_only, 1), 0,
                 "not hard-failing per documented advisory-only policy", "::error::"},
                {"mixed-advisory-nonadvisory", stub_write_report(json_mixed, 1), 1,
                 "Shim security modules FAILED", nullptr},
                {"clean-pass-rc0", stub_write_report(json_all_pass, 0), 0,
                 "Shim gate passed", nullptr},
                {"nonzero-no-classified-cause", stub_write_report(json_all_pass, 1), 1,
                 "unexplained hard failure", nullptr},
                {"gpu-advisory-false-pass", stub_write_report(json_gpu_false_pass, 0), 1,
                 "Rule 16 violation", nullptr},
            };

            for (const auto& sc : scenarios) {
                bool ok = run_scenario(dedented, scratch_root, sc);
                CHECK(ok, (std::string("executable scenario: ") + sc.name).c_str());
            }

            fs::remove_all(scratch_root, ec);
        }
    }

    printf("  [shim-gate-report-policy] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail;
}

#if defined(STANDALONE_TEST) || !defined(UNIFIED_AUDIT_RUNNER)
int main() { return test_shim_security_gate_policy_run(); }
#endif
