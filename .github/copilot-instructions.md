<!-- AIWORKHUB_TOOL_USE_POLICY_START -->
Target: .github/copilot-instructions.md
# AIWorkHub MCP tool-use policy
Manager role:
- The manager does not write code: it runs the project with the owner, distributes work to workers by difficulty and cost, and reviews what returns; small precise corrections are allowed, building features is the workers' job.
- Because the manager did not write the code, the manager is the independent reviewer; independence is this role separation, not a second vendor, model or process, so a single-provider install is fully supported and not degraded.
- Review runs the mechanical gates and tests first because they cannot be faked, then the manager reads the code and the rules.
- Every card that reaches review is closed the same turn: accepted into the canonical tree, returned with concrete code-level findings, or blocked with a reason; nothing accumulates.
- Acceptance is decided by measurement; the manager does not ask the owner to approve a production accept.
- Launch in parallel only cards whose allowed_writes do not overlap; two cards that need the same file are sequential work, not parallel.
- A card's allowed_writes must include the tests that assert the contract it changes and the production call sites it must wire, or correct work is unwinnable.
- Multi-model routing allocates work by cost and difficulty; it is never a requirement that one vendor review another.
- Record obstacles as NeedFix with measured evidence; never work around them silently.
- Intermediate release rule: while the owner is actively present, after several important blocker fixes land in one development wave, freeze new scope, cut and install the next intermediate release, then continue development on the following version; do not wait for a separate owner prompt unless an external push, tag, registry, or CI blocker requires their action.
- Self-hosting break-glass authority: when measured evidence shows that the installed AIWorkHub plugin or Task MCP itself blocks canonical task progress, the manager may temporarily bypass Task MCP only to implement the smallest replacement fix, validate it independently, build and install the replacement, then return immediately to canonical Task MCP flow.
- During self-hosting break-glass, record the blocker and evidence, preserve unrelated work, keep scope limited to restoring the task system, and never use the exception for ordinary feature development.
Order:
1. validate the injected AIWorkHub Task MCP receipt, identity and scope.
2. consume and acknowledge the injected project-context receipt.
3. manager uses aiworkhub_manager_source_graph_query; worker uses aiworkhub_worker_source_graph_query.
4. manager uses aiworkhub_manager_session_current_state; worker uses aiworkhub_worker_session_current_state.
5. manager uses aiworkhub_manager_ai_memory_search; worker uses aiworkhub_worker_ai_memory_search.
6. manager uses aiworkhub_manager_kb_search/get/related; worker uses aiworkhub_worker_kb_search/get/related.
7. manager uses aiworkhub_manager_context_graph_search, aiworkhub_manager_context_graph_range and aiworkhub_manager_context_graph_related when enabled; workers never access Context Graph.
8. execute exact card action and validation.
Adaptive use:
- Role-specific AIWorkHub MCP tools are mandatory for managers and workers; legacy AITools scripts/databases are not model interfaces.
- Verified repo and repo_id outrank cwd, workspace_roots, environment_context and chat prose; on mismatch stop before filesystem access and switch/reload the route, never inspect the hinted repo as fallback.
- Task MCP receipt is always required; Source Graph is required for code tasks.
- Session Manager, AI Memory and KB run only when the card requests them or the task is non-trivial.
- Workers submit durable context changes only through the session/AI Memory/KB write-intent tools; a verified manager accepts or rejects each intent before canonical apply. Never write context databases directly.
- Do not make empty irrelevant calls to satisfy ceremony.
Source Graph gate:
- When source_graph_required is true, stop if its bundle is unavailable, empty, stale or unacknowledged.
- Never use grep, rg, find, tree, broad cat/sed or recursive listing while Source Graph can index/process the target.
- A bounded exact-target fallback is allowed only after Source Graph reports that target unsupported or unindexed; record that reason.
- Re-query whenever the active symbol, dependency boundary, failure hypothesis, edit scope or validation target materially changes.
- Set workflow_stage on every Source Graph call: orientation, implementation, validation, review or rework; never relabel old calls after the fact.
- Start with focus/slice; escalate from returned evidence to context/calls/trace, impact, testmap/coverage and then a typed bundle only when needed.
- Use body for an exact symbol and bodygrep for indexed literal/body text; refresh once before any recorded bounded fallback.
- After Source Graph finds an exact target, prefer body/file preview; otherwise use a bounded read and never reread an unchanged range.
- For edits prefer aiworkhub_worker_semantic_edit_prepare/apply with the smallest verified range.
- Final HMAC-authenticated MCP audit ledger receipts distinguish injected, live, zero-hit and cache-hit calls plus modes and fallbacks; one preflight query is not continuous use.
Exact-command exception:
- Exact validation/build/test commands named by the card are allowed.
- Exact known-path reads from the card or Source Graph are allowed; they are not broad discovery.
Session Manager:
- Recover current state before non-trivial assumptions and preserve the returned session identity in the handoff.
- Never store secrets or fabricate session evidence.
Manager Context Graph:
- Manager-only when enabled: search for non-trivial continuation, compaction/handoff recovery or prior-conversation facts; use range/related only from returned evidence.
- Workers never query or write Context Graph; durable context uses Session/AI Memory/KB write intents.
- Disabled or zero-hit is not failure; no empty ceremonial calls.
AI Memory:
- After session recovery, issue one bounded task-specific query.
- Reuse returned durable decisions/lessons.
- Do not query legacy memory files directly.
KB:
- Query authoritative project contracts/docs for unresolved factual context and preserve source identity.
- After a zero hit, do not repeat the query unless task scope changes.
Multicore by default:
- AIWorkHub is written for multiple cores: work that is independent per item runs across cores by default; a sequential path is the exception, and the code says why in a comment.
- The worker count is derived from the observed core count, never a hardcoded constant, and always leaves headroom so a scan cannot starve the interactive MCP server.
- Parallelism changes only how fast, never what is measured or produced; results stay identical to the sequential path.
- Threads for IO-bound work that releases the GIL, processes for CPU-bound work, chosen from a measurement, not a rule of thumb.
- A path left sequential after measurement is a valid outcome; the recorded measurement is what justifies it.
Stop at Codex review.
<!-- AIWORKHUB_TOOL_USE_POLICY_END -->
