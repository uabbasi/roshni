"""User-facing workflow tools — follows create_task_tools() pattern.

Returns list[ToolDefinition] for integration into the agent's tool set.
Tools are gated on workflow.enabled config flag.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from roshni.agent.tools import ToolDefinition

if TYPE_CHECKING:
    from .orchestrator import Orchestrator
    from .store import ProjectStore


def _fmt_project(p) -> str:
    """Format a Project for tool output.

    Shows workflow details (budget, phases) only for managed projects.
    Obsidian-only projects get a simpler format.
    """
    lines = [
        f"**{p.id}** — {p.goal[:80]}",
        f"Status: {p.status.value}",
    ]
    # Show workflow details only for managed projects (those with phases or a plan)
    if p.phases or p.plan_hash:
        lines.append(
            f"Budget: ${p.budget.cost_used_usd:.2f}/${p.budget.max_cost_usd:.2f} USD | "
            f"{p.budget.llm_calls_used}/{p.budget.max_llm_calls} calls"
        )
        lines.append(
            f"Phases: {len(p.phases)} ({sum(1 for ph in p.phases if ph.status.value == 'completed')} completed)"
        )
    if p.tags:
        lines.append(f"Tags: {', '.join(p.tags)}")
    if p.artifacts:
        lines.append(f"Artifacts: {', '.join(a.name for a in p.artifacts)}")

    # Recent journal (last 3)
    if p.journal:
        lines.append("\nRecent activity:")
        for entry in p.journal[-3:]:
            lines.append(f"  [{entry.actor}] {entry.action}: {entry.content[:80]}")

    return "\n".join(lines)


def _fmt_project_detail(p) -> str:
    """Format a Project with full detail."""
    lines = [_fmt_project(p), ""]

    if p.phases:
        lines.append("## Phases")
        for phase in p.phases:
            lines.append(f"\n### {phase.id}: {phase.name} [{phase.status.value}]")
            if phase.description:
                lines.append(f"  {phase.description}")
            if phase.tasks:
                for task in phase.tasks:
                    lines.append(f"  - {task.id}: {task.description}")

    if p.terminal_conditions:
        lines.append("\n## Terminal Conditions")
        for tc in p.terminal_conditions:
            check = "met" if tc.met else "not met"
            lines.append(f"  - [{check}] {tc.description} ({tc.type})")

    return "\n".join(lines)


def create_workflow_tools(
    store: ProjectStore,
    orchestrator: Orchestrator,
    send_fn: Callable | None = None,
) -> list[ToolDefinition]:
    """Create user-facing workflow tools."""
    import asyncio

    # Capture the main event loop for cross-thread scheduling.
    # Tool functions execute inside DefaultAgent.chat() which runs in a thread
    # pool executor via invoke(). Without this, asyncio.run() would create a
    # new event loop where Telegram's httpx client (bound to the main loop) fails.
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        _main_loop = None

    def _run_async(coro):
        """Run an async coroutine from sync context (tool execution thread)."""
        # Prefer scheduling on the main loop — avoids creating a new loop
        # that would break Telegram's httpx client and other loop-bound resources.
        if _main_loop is not None and _main_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, _main_loop)
            return future.result(timeout=300)
        # Fallback: no main loop captured (e.g. testing)
        return asyncio.run(coro)

    # -- create_project ------------------------------------------------------

    def create_project(goal: str, max_cost_usd: float = 5.0, max_llm_calls: int = 100, tags: str = "") -> str:
        """Create a new long-running project from a goal description.

        The orchestrator will decompose the goal into phases and tasks.
        The project starts in PLANNING status and must be approved before execution.
        """
        from .models import Budget

        budget = Budget(max_cost_usd=max_cost_usd, max_llm_calls=max_llm_calls)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        try:
            project = _run_async(orchestrator.start_project(goal, budget, tags=tag_list))
            return f"Project created and planned.\n\n{_fmt_project_detail(project)}"
        except Exception as e:
            return f"Error creating project: {e}"

    # -- check_project -------------------------------------------------------

    def check_project(project_id: str = "") -> str:
        """Check status of one or all projects.

        If project_id is empty, lists all projects.
        If project_id is provided, shows detailed status.
        """
        try:
            if project_id:
                project = _run_async(store.get(project_id))
                if project is None:
                    return f"Project not found: {project_id}"
                return _fmt_project_detail(project)
            else:
                projects = _run_async(store.list_projects())
                if not projects:
                    return "No projects found."
                return "\n\n---\n\n".join(_fmt_project(p) for p in projects)
        except Exception as e:
            return f"Error checking project: {e}"

    # -- steer_project -------------------------------------------------------

    def steer_project(project_id: str, direction: str) -> str:
        """Inject user guidance into a running project."""
        try:
            _run_async(orchestrator.steer(project_id, direction))
            return f"Steering applied to {project_id}: {direction[:100]}"
        except Exception as e:
            return f"Error steering project: {e}"

    # -- approve_project -----------------------------------------------------

    def approve_project(project_id: str) -> str:
        """Approve a project plan and begin execution."""
        import asyncio

        try:
            # Fire-and-forget: launch execution in the background so the tool
            # returns immediately.  The orchestrator reports progress via send_fn
            # (Telegram messages at each phase boundary).
            if _main_loop is not None and _main_loop.is_running():

                async def _run_and_report():
                    try:
                        await orchestrator.approve_and_execute(project_id)
                    except Exception as exc:
                        logger.error(f"Background execution failed for {project_id}: {exc}")
                        if orchestrator._send_fn:
                            try:
                                await orchestrator._send_fn(f"Project {project_id} execution failed: {exc}")
                            except Exception:
                                pass

                asyncio.run_coroutine_threadsafe(_run_and_report(), _main_loop)
                return (
                    f"Project {project_id} approved. Execution started in the background — "
                    f"I'll report progress as each phase completes."
                )
            else:
                # Fallback: blocking execution (e.g. testing without event loop)
                _run_async(orchestrator.approve_and_execute(project_id))
                return f"Project {project_id} approved and execution completed."
        except Exception as e:
            return f"Error approving project: {e}"

    # -- pause_project -------------------------------------------------------

    def pause_project(project_id: str) -> str:
        """Pause a running project. Active workers will drain before pausing."""
        try:
            project = _run_async(store.transition(project_id, "paused", actor="user"))
            return f"Project {project_id} pause requested. Status: {project.status.value}"
        except Exception as e:
            return f"Error pausing project: {e}"

    # -- resume_project ------------------------------------------------------

    def resume_project(project_id: str) -> str:
        """Resume a paused project."""
        try:
            project = _run_async(store.get(project_id))
            if project is None:
                return f"Project not found: {project_id}"
            if project.status.value != "paused":
                return f"Project {project_id} is {project.status.value}, not paused"
            project = _run_async(store.transition(project_id, "executing", actor="user"))
            # Re-trigger execution
            _run_async(orchestrator.approve_and_execute(project_id))
            return f"Project {project_id} resumed."
        except Exception as e:
            return f"Error resuming project: {e}"

    # -- cancel_project ------------------------------------------------------

    def cancel_project(project_id: str) -> str:
        """Cancel a project. This is terminal — the project cannot be restarted."""
        try:
            _run_async(store.transition(project_id, "cancelled", actor="user"))
            return f"Project {project_id} cancelled."
        except Exception as e:
            return f"Error cancelling project: {e}"

    # -- reconcile_project ---------------------------------------------------

    def reconcile_project(project_id: str, accept_obsidian: bool = False) -> str:
        """Resolve an Obsidian conflict.

        If accept_obsidian=True, accepts structured human edits from Obsidian.
        Otherwise, overrides Obsidian with the canonical plan.
        """
        try:
            _run_async(orchestrator.reconcile(project_id, accept_obsidian))
            strategy = "accepted human edits" if accept_obsidian else "overrode with canonical plan"
            return f"Conflict resolved for {project_id}: {strategy}"
        except Exception as e:
            return f"Error reconciling project: {e}"

    # -- advance_project -----------------------------------------------------

    def advance_project(project_id: str, directive: str = "") -> str:
        """Advance an existing project — add new work, continue progress.

        Works on done, reviewing, paused, or executing projects.
        For done/reviewing: creates a new phase and executes it.
        For paused: resumes execution.
        For executing: injects guidance.
        """
        try:
            if _main_loop is not None and _main_loop.is_running():

                async def _advance_and_report():
                    try:
                        await orchestrator.advance(project_id, directive)
                    except Exception as exc:
                        logger.error(f"Advance failed for {project_id}: {exc}")
                        if orchestrator._send_fn:
                            try:
                                await orchestrator._send_fn(f"Project {project_id} advance failed: {exc}")
                            except Exception:
                                pass

                asyncio.run_coroutine_threadsafe(_advance_and_report(), _main_loop)
                action = f" with directive: {directive[:80]}" if directive else ""
                return f"Advancing project {project_id}{action}. Working in the background — I'll report progress."
            else:
                _run_async(orchestrator.advance(project_id, directive))
                return f"Project {project_id} advanced."
        except Exception as e:
            return f"Error advancing project: {e}"

    # -- review_projects -----------------------------------------------------

    def review_projects(query: str = "", tags: str = "") -> str:
        """Cross-project research and synthesis.

        Analyzes findings, connections, and recommended actions across
        matching projects. Filter by query text or comma-separated tags.
        """
        try:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
            result = _run_async(orchestrator.review_projects(query, tag_list))
            return result
        except Exception as e:
            return f"Error reviewing projects: {e}"

    # -- Build tool list -----------------------------------------------------

    tools: list[ToolDefinition] = [
        ToolDefinition(
            name="create_project",
            description=(
                "Create a new long-running project from a goal. The orchestrator decomposes "
                "the goal into phases with tasks. Returns the plan for approval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The project goal to achieve"},
                    "max_cost_usd": {"type": "number", "description": "Maximum budget in USD", "default": 5.0},
                    "max_llm_calls": {"type": "integer", "description": "Maximum LLM calls", "default": 100},
                    "tags": {"type": "string", "description": "Comma-separated tags", "default": ""},
                },
                "required": ["goal"],
            },
            function=create_project,
            permission="write",
        ),
        ToolDefinition(
            name="check_project",
            description="Check status of one or all workflow projects. Empty project_id lists all.",
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID (empty for all)", "default": ""},
                },
            },
            function=check_project,
            permission="read",
        ),
        ToolDefinition(
            name="steer_project",
            description="Inject user guidance into a running project to adjust its direction.",
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID"},
                    "direction": {"type": "string", "description": "Guidance for the project"},
                },
                "required": ["project_id", "direction"],
            },
            function=steer_project,
            permission="write",
        ),
        ToolDefinition(
            name="approve_project",
            description="Approve a project plan and begin execution.",
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID to approve"},
                },
                "required": ["project_id"],
            },
            function=approve_project,
            permission="write",
        ),
        ToolDefinition(
            name="pause_project",
            description="Pause a running project. Active workers drain before pausing.",
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID to pause"},
                },
                "required": ["project_id"],
            },
            function=pause_project,
            permission="write",
        ),
        ToolDefinition(
            name="resume_project",
            description="Resume a paused project.",
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID to resume"},
                },
                "required": ["project_id"],
            },
            function=resume_project,
            permission="write",
        ),
        ToolDefinition(
            name="cancel_project",
            description="Cancel a project (terminal). Cannot be undone.",
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID to cancel"},
                },
                "required": ["project_id"],
            },
            function=cancel_project,
            permission="write",
            requires_approval=True,
        ),
        ToolDefinition(
            name="reconcile_project",
            description="Resolve an Obsidian conflict. accept_obsidian=true to accept human edits.",
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID"},
                    "accept_obsidian": {
                        "type": "boolean",
                        "description": "True to accept human edits, false to override",
                        "default": False,
                    },
                },
                "required": ["project_id"],
            },
            function=reconcile_project,
            permission="write",
        ),
        ToolDefinition(
            name="advance_project",
            description=(
                "Advance an existing project — continue work, add new phases, or resume progress. "
                "Use for long-running projects that need periodic attention. "
                "Provide a directive to focus the new work, or leave empty to continue naturally."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID to advance"},
                    "directive": {
                        "type": "string",
                        "description": "What to focus on (empty = continue naturally)",
                        "default": "",
                    },
                },
                "required": ["project_id"],
            },
            function=advance_project,
            permission="write",
        ),
        ToolDefinition(
            name="review_projects",
            description=(
                "Cross-project research and synthesis. Analyzes themes, progress, and "
                "recommended actions across projects. Filter by query or tags."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to filter projects (matches goal, phase names, tags)",
                        "default": "",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags to filter projects",
                        "default": "",
                    },
                },
            },
            function=review_projects,
            permission="read",
        ),
    ]

    return tools
