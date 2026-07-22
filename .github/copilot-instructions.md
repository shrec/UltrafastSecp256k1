<!-- AIWORKHUB_TOOL_USE_POLICY_START -->
Target: .github/copilot-instructions.md
# AIWorkHub MCP tool-use policy
Order:
1. validate the injected AIWorkHub Task MCP receipt, identity and scope.
2. consume and acknowledge the injected project-context receipt.
3. manager uses aiworkhub_manager_source_graph_query; worker uses aiworkhub_worker_source_graph_query.
4. manager uses aiworkhub_manager_session_current_state; worker uses aiworkhub_worker_session_current_state.
5. manager uses aiworkhub_manager_ai_memory_search; worker uses aiworkhub_worker_ai_memory_search.
6. manager uses aiworkhub_manager_kb_search/get/related; worker uses aiworkhub_worker_kb_search/get/related.
7. execute exact card action and validation.
Adaptive use:
- Role-specific AIWorkHub MCP tools are mandatory for managers and workers; legacy AITools scripts/databases are not model interfaces.
- Task MCP receipt is always required; Source Graph is required for code tasks.
- Session Manager, AI Memory and KB run only when the card requests them or the task is non-trivial.
- Do not make empty irrelevant calls to satisfy ceremony.
Source Graph gate:
- When source_graph_required is true, stop if its bundle is unavailable, empty, stale or unacknowledged.
- Never use grep, rg, find, tree, broad cat/sed or recursive listing while Source Graph can index/process the target.
- A bounded exact-target fallback is allowed only after Source Graph reports that target unsupported or unindexed; record that reason.
Exact-command exception:
- Exact validation/build/test commands named by the card are allowed.
- Exact known-path reads from the card or Source Graph are allowed; they are not broad discovery.
Session Manager:
- Recover current state before non-trivial assumptions and preserve the returned session identity in the handoff.
- Never store secrets or fabricate session evidence.
AI Memory:
- After session recovery, issue one bounded task-specific query.
- Reuse returned durable decisions/lessons.
- Do not query legacy memory files directly.
KB:
- Query authoritative project contracts/docs for unresolved factual context and preserve source identity.
- After a zero hit, do not repeat the query unless task scope changes.
Stop at Codex review.
<!-- AIWORKHUB_TOOL_USE_POLICY_END -->
