"""Orchestrator — the brain of the workflow system.

Composes (not inherits) a DefaultAgent for planning decisions.
Uses thinking-tier model for planning, light model for workers.
Reports to user via send_fn callback at phase boundaries.

Internal tools (NOT exposed to Hakim):
    decompose_goal, evaluate_condition, advance_phase,
    spawn_worker, report_to_user, update_plan
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from .backend import FileWorkflowBackend
from .events import (
    BUDGET_EXHAUSTED,
    BUDGET_WARNING,
    PHASE_COMPLETED,
    PHASE_FAILED,
    PHASE_STARTED,
    PLAN_WRITTEN,
    PROJECT_ADVANCED,
    PROJECT_TRANSITIONED,
    TERMINAL_CONDITION_EVALUATED,
)
from .models import (
    Budget,
    JournalEntry,
    Phase,
    PhaseEntry,
    PhaseStatus,
    Project,
    ProjectStatus,
    TaskSpec,
    TerminalCondition,
    compute_plan_hash,
    validate_transition,
)
from .store import ProjectStore
from .worker import WorkerPool

if TYPE_CHECKING:
    from roshni.agent.tools import ToolDefinition
    from roshni.core.config import Config
    from roshni.core.events import EventBus
    from roshni.core.llm.model_selector import ModelSelector
    from roshni.core.secrets import SecretsManager

SendFn = Callable[[str], Any]  # async or sync callback to send messages to user


class Orchestrator:
    """The orchestrator decomposes goals, manages phases, and evaluates terminal conditions.

    Composition pattern: wraps a DefaultAgent for planning decisions,
    delegates task execution to WorkerPool.
    """

    def __init__(
        self,
        config: Config,
        secrets: SecretsManager,
        store: ProjectStore,
        backend: FileWorkflowBackend,
        event_bus: EventBus | None = None,
        model_selector: ModelSelector | None = None,
        send_fn: SendFn | None = None,
        worker_pool: WorkerPool | None = None,
        all_tools: list[ToolDefinition] | None = None,
    ) -> None:
        self._config = config
        self._secrets = secrets
        self._store = store
        self._backend = backend
        self._event_bus = event_bus
        self._model_selector = model_selector
        self._send_fn = send_fn
        self._worker_pool = worker_pool
        self._all_tools = all_tools or []

    # -- Public API ----------------------------------------------------------

    async def start_project(self, goal: str, budget: Budget | None = None, tags: list[str] | None = None) -> Project:
        """Phase 0: Create a project and generate the initial plan.

        Creates the project in PLANNING status, then uses a thinking-tier
        model to decompose the goal into phases with tasks.
        """
        project = await self._store.create(goal=goal, budget=budget, tags=tags)
        await self._report(f"Project created: {project.id}\nGoal: {goal}\nStatus: Planning...")

        # Use LLM to decompose goal into phases
        try:
            await self._decompose_goal(project)
            # Save plan
            await self._backend.save_plan(project)
            plan_hash = compute_plan_hash(project)
            project.plan_hash = plan_hash

            # Record plan event
            evt = self._backend.create_event(project.id, PLAN_WRITTEN, "orchestrator", {"plan_hash": plan_hash})
            await self._backend.record_event(project.id, evt)
            project.last_event_seq = evt.seq

            # Transition to awaiting approval
            await self._transition(project, ProjectStatus.AWAITING_APPROVAL)

            await self._report(
                f"Plan ready for project {project.id}.\n"
                f"Phases: {len(project.phases)}\n"
                f"Use approve_project to begin execution."
            )
        except Exception as e:
            logger.error(f"Failed to plan project {project.id}: {e}")
            project.journal.append(
                JournalEntry(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    actor="orchestrator",
                    action="error",
                    content=f"Planning failed: {e}",
                )
            )
            await self._transition(project, ProjectStatus.FAILED)

        await self._store.update(project)
        return project

    async def approve_and_execute(self, project_id: str) -> None:
        """Approve the plan and begin execution."""
        project = await self._store.get(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        if project.status != ProjectStatus.AWAITING_APPROVAL:
            raise ValueError(f"Project {project_id} is {project.status.value}, not awaiting_approval")

        await self._transition(project, ProjectStatus.EXECUTING)
        await self._report(f"Project {project_id} approved. Beginning execution...")

        # Execute phases sequentially
        try:
            await self._execute_phases(project)
        except Exception as e:
            logger.error(f"Execution failed for {project_id}: {e}")
            project.journal.append(
                JournalEntry(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    actor="orchestrator",
                    action="error",
                    content=f"Execution failed: {e}",
                )
            )
            if project.status not in {ProjectStatus.PAUSED, ProjectStatus.CANCELLED}:
                await self._transition(project, ProjectStatus.FAILED)
            await self._store.update(project)

    async def steer(self, project_id: str, direction: str) -> None:
        """Inject user guidance into a running project."""
        project = await self._store.get(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        project.journal.append(
            JournalEntry(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                actor="user",
                action="steering",
                content=direction,
            )
        )

        from .events import PROJECT_STEERED

        evt = self._backend.create_event(project_id, PROJECT_STEERED, "user", {"direction": direction})
        await self._backend.record_event(project_id, evt)
        project.last_event_seq = evt.seq

        await self._store.update(project)
        logger.info(f"Steering applied to {project_id}: {direction[:60]}")

    async def reconcile(self, project_id: str, accept_obsidian: bool = False) -> None:
        """Resolve an Obsidian conflict."""
        project = await self._store.get(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        if accept_obsidian:
            await self._backend.reconcile_accept_obsidian(project)
            strategy = "accept_obsidian"
        else:
            await self._backend.reconcile_override_obsidian(project)
            strategy = "override"

        from .events import CONFLICT_RECONCILED

        evt = self._backend.create_event(project_id, CONFLICT_RECONCILED, "user", {"strategy": strategy})
        await self._backend.record_event(project_id, evt)
        project.last_event_seq = evt.seq

        project.journal.append(
            JournalEntry(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                actor="user",
                action="reconcile",
                content=f"Conflict resolved: {strategy}",
            )
        )

        await self._store.update(project)
        logger.info(f"Conflict reconciled for {project_id}: {strategy}")

    async def advance(self, project_id: str, directive: str = "") -> Project:
        """Re-open a completed/reviewing project and add new work.

        For done/reviewing projects: transitions to planning, creates new
        phase(s) based on directive + existing context, then auto-executes.
        For paused projects: resumes + optionally steers.
        For executing projects: just steers.
        """
        project = await self._store.get(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        if project.status in {ProjectStatus.DONE, ProjectStatus.REVIEWING}:
            # Re-open: plan new phase(s) based on what's been done + directive
            await self._transition(project, ProjectStatus.PLANNING)
            await self._report(f"Advancing project {project_id}...")

            try:
                new_phase = await self._plan_advance(project, directive)

                # Record advance event
                evt = self._backend.create_event(
                    project.id,
                    PROJECT_ADVANCED,
                    "orchestrator",
                    {"directive": directive[:200], "new_phase_id": new_phase.id},
                )
                await self._backend.record_event(project.id, evt)
                project.last_event_seq = evt.seq

                # Save updated plan
                await self._backend.save_plan(project)
                plan_hash = compute_plan_hash(project)
                project.plan_hash = plan_hash

                evt = self._backend.create_event(project.id, PLAN_WRITTEN, "orchestrator", {"plan_hash": plan_hash})
                await self._backend.record_event(project.id, evt)
                project.last_event_seq = evt.seq

                # Auto-approve and execute (user explicitly asked to advance)
                await self._transition(project, ProjectStatus.AWAITING_APPROVAL)
                await self._transition(project, ProjectStatus.EXECUTING)
                await self._store.update(project)

                await self._report(f"New phase added: {new_phase.name}. Executing...")
                await self._execute_phases(project)

            except Exception as e:
                logger.error(f"Failed to advance project {project_id}: {e}")
                project.journal.append(
                    JournalEntry(
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        actor="orchestrator",
                        action="error",
                        content=f"Advance failed: {e}",
                    )
                )
                if project.status not in {ProjectStatus.PAUSED, ProjectStatus.CANCELLED}:
                    await self._transition(project, ProjectStatus.FAILED)
                await self._store.update(project)

        elif project.status == ProjectStatus.PAUSED:
            # Resume paused project
            await self._transition(project, ProjectStatus.EXECUTING)
            if directive:
                await self.steer(project_id, directive)
            await self._store.update(project)
            await self._report(f"Project {project_id} resumed.")
            await self._execute_phases(project)

        elif project.status == ProjectStatus.EXECUTING:
            # Already running — just steer
            if directive:
                await self.steer(project_id, directive)
                await self._report(f"Steering applied to running project {project_id}.")
            else:
                await self._report(f"Project {project_id} is already executing.")

        else:
            raise ValueError(
                f"Cannot advance project {project_id} in status {project.status.value}. "
                f"Only done, reviewing, paused, or executing projects can be advanced."
            )

        return await self._store.get(project_id)

    async def review_projects(self, query: str = "", tags: list[str] | None = None) -> str:
        """Cross-project synthesis — analyze findings across multiple projects.

        Returns an LLM-generated synthesis of matching projects.
        """
        projects = await self._store.list_projects()
        if not projects:
            return "No projects found."

        # Filter by tags if provided
        if tags:
            tag_set = set(tags)
            projects = [p for p in projects if tag_set & set(p.tags)]

        # Filter by query (simple keyword match on goal + phase names)
        if query:
            query_lower = query.lower()
            projects = [
                p
                for p in projects
                if query_lower in p.goal.lower()
                or any(query_lower in ph.name.lower() for ph in p.phases)
                or any(query_lower in t for t in p.tags)
            ]

        if not projects:
            return f"No projects matching query='{query}' tags={tags}."

        # Build context for synthesis
        context_parts = []
        for p in projects:
            completed = sum(1 for ph in p.phases if ph.status == PhaseStatus.COMPLETED)
            part = (
                f"### {p.id}: {p.goal}\n"
                f"Status: {p.status.value} | Tags: {', '.join(p.tags) if p.tags else 'none'}\n"
                f"Phases: {completed}/{len(p.phases)} completed\n"
            )
            if p.artifacts:
                part += f"Artifacts: {', '.join(a.name for a in p.artifacts)}\n"
            # Include recent journal entries for context
            if p.journal:
                recent = p.journal[-3:]
                part += "Recent activity:\n"
                for j in recent:
                    part += f"  - [{j.actor}] {j.action}: {j.content[:100]}\n"
            # Include unmet terminal conditions
            unmet = [tc for tc in p.terminal_conditions if not tc.met]
            if unmet:
                part += "Unmet conditions:\n"
                for tc in unmet:
                    part += f"  - {tc.description}\n"
            context_parts.append(part)

        context = "\n---\n".join(context_parts)

        # Run LLM synthesis
        from roshni.agent.default import DefaultAgent

        reviewer = DefaultAgent(
            self._config,
            self._secrets,
            tools=[],
            name="project-reviewer",
            model_selector=self._model_selector,
        )

        prompt = (
            f"You are a personal project advisor. Analyze these projects and provide:\n"
            f"1. Cross-cutting themes and connections\n"
            f"2. Key findings and progress summary\n"
            f"3. Recommended next actions (prioritized)\n"
            f"4. Any risks or stalled areas\n\n"
            f"{'Query: ' + query + chr(10) if query else ''}"
            f"## Projects\n\n{context}\n\n"
            f"Be concise and actionable. Focus on insights the user might miss."
        )

        try:
            result = await reviewer.invoke(prompt, channel="workflow")
            return result.strip()
        except Exception as e:
            logger.error(f"Review synthesis failed: {e}")
            return f"Error generating review: {e}"

    # -- Internal: planning --------------------------------------------------

    async def _decompose_goal(self, project: Project) -> None:
        """Use LLM to break down a goal into phases with tasks.

        In v1, this creates a simple structured plan. Future versions
        will use the thinking-tier model for richer decomposition.
        """
        from roshni.agent.default import DefaultAgent

        planning_system_prompt = (
            "You are a project planner for a personal AI assistant. Your job is to decompose "
            "a user's goal into a structured execution plan with sequential phases.\n\n"
            "## Planning Principles\n"
            "- Each phase should represent a distinct stage of work (research, analysis, synthesis, etc.)\n"
            "- Tasks within a phase are the concrete actions a worker agent will execute\n"
            "- Workers have access to tools: web search, file operations, note-taking, calculations\n"
            "- Keep phases focused — 1-3 tasks each is ideal\n"
            "- Entry criteria define what must be true before a phase starts\n"
            "- Exit criteria define what must be true for a phase to be considered complete\n"
            "- Be specific in task descriptions — workers execute them literally\n\n"
            "## Output Format\n"
            "Respond with a JSON object (no markdown fencing, no explanation):\n"
            '{"phases": [{"id": "phase-1", "name": "...", "description": "...", '
            '"tasks": [{"id": "task-001", "description": "..."}], '
            '"entry_criteria": ["..."], "exit_criteria": ["..."]}]}\n\n'
            "## Example\n"
            "Goal: Research the best home solar panel options and write a comparison\n\n"
            '{"phases": [\n'
            '  {"id": "phase-1", "name": "Research",\n'
            '   "description": "Gather info on top solar panel brands",\n'
            '   "tasks": [\n'
            '    {"id": "task-001", "description": "Search for top 5 residential '
            'solar panel manufacturers in 2026"},\n'
            '    {"id": "task-002", "description": "Search for recent reviews '
            'comparing residential solar panels"}\n'
            '   ], "entry_criteria": [], "exit_criteria": ["5+ manufacturers identified"]},\n'
            '  {"id": "phase-2", "name": "Analysis",\n'
            '   "description": "Compare options across key dimensions",\n'
            '   "tasks": [\n'
            '    {"id": "task-003", "description": "Create comparison matrix: '
            'efficiency, cost/watt, warranty, degradation"}\n'
            '   ], "entry_criteria": ["Research done"], '
            '"exit_criteria": ["Matrix complete"]},\n'
            '  {"id": "phase-3", "name": "Synthesis",\n'
            '   "description": "Write the final recommendation",\n'
            '   "tasks": [\n'
            '    {"id": "task-004", "description": "Write 500-word summary with '
            'top 3 picks, pros/cons, and a winner"}\n'
            '   ], "entry_criteria": ["Analysis complete"], '
            '"exit_criteria": ["Summary written"]}\n'
            "]}\n"
        )

        planning_prompt = f"Goal: {project.goal}\n\nDecompose this into 2-5 sequential phases. Respond with ONLY JSON."

        planner = DefaultAgent(
            self._config,
            self._secrets,
            tools=[],
            name="orchestrator-planner",
            system_prompt=planning_system_prompt,
            model_selector=self._model_selector,
        )

        result = await planner.invoke(planning_prompt, channel="workflow")

        # Parse the structured plan
        try:
            # Try to extract JSON from the response
            plan_text = result.strip()
            # Remove markdown code fencing if present
            if plan_text.startswith("```"):
                lines = plan_text.split("\n")
                plan_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            plan = json.loads(plan_text)
            phases_data = plan.get("phases", [])

            for i, pd in enumerate(phases_data):
                phase_id = pd.get("id", f"phase-{i + 1}")
                tasks = []
                for j, td in enumerate(pd.get("tasks", [])):
                    task_id = td.get("id", f"task-{(i * 100) + j + 1:03d}")
                    tasks.append(
                        TaskSpec(
                            id=task_id,
                            description=td.get("description", ""),
                            allowed_tools=td.get("allowed_tools", []),
                        )
                    )

                entry_criteria = [PhaseEntry(description=ec) for ec in pd.get("entry_criteria", [])]
                exit_criteria = [PhaseEntry(description=ec) for ec in pd.get("exit_criteria", [])]

                project.phases.append(
                    Phase(
                        id=phase_id,
                        name=pd.get("name", f"Phase {i + 1}"),
                        description=pd.get("description", ""),
                        tasks=tasks,
                        entry_criteria=entry_criteria,
                        exit_criteria=exit_criteria,
                    )
                )

            project.journal.append(
                JournalEntry(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    actor="orchestrator",
                    action="planned",
                    content=f"Decomposed goal into {len(project.phases)} phases",
                )
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse plan JSON: {e}")
            # Create a single-phase fallback
            project.phases.append(
                Phase(
                    id="phase-1",
                    name="Execute goal",
                    description=project.goal,
                    tasks=[TaskSpec(id="task-001", description=project.goal)],
                )
            )
            project.journal.append(
                JournalEntry(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    actor="orchestrator",
                    action="planned",
                    content="Fallback: created single-phase plan (LLM output not parseable)",
                )
            )

    async def _plan_advance(self, project: Project, directive: str) -> Phase:
        """Plan new phase(s) for advancing an existing project.

        Uses LLM with full project context to create targeted next steps.
        Returns the newly created Phase.
        """
        from roshni.agent.default import DefaultAgent

        # Build context of what's been done
        completed_phases = [
            f"- {ph.name}: {ph.description}" for ph in project.phases if ph.status == PhaseStatus.COMPLETED
        ]
        artifacts_list = [f"- {a.name} ({a.mime_type})" for a in project.artifacts]
        unmet_conditions = [tc.description for tc in project.terminal_conditions if not tc.met]

        # Calculate next phase/task IDs
        next_phase_num = len(project.phases) + 1
        next_task_start = sum(len(ph.tasks) for ph in project.phases) + 1

        context = (
            f"You are advancing an existing project. Plan the NEXT phase of work.\n\n"
            f"## Project\n"
            f"Goal: {project.goal}\n"
            f"Status: {project.status.value}\n\n"
            f"## Completed phases\n"
            f"{chr(10).join(completed_phases) if completed_phases else 'None yet'}\n\n"
            f"## Artifacts produced\n"
            f"{chr(10).join(artifacts_list) if artifacts_list else 'None yet'}\n\n"
        )
        if unmet_conditions:
            context += f"## Unmet conditions to address\n{chr(10).join('- ' + c for c in unmet_conditions)}\n\n"

        if directive:
            context += f"## User directive\n{directive}\n\n"
        else:
            context += "## User directive\nContinue making progress on this project.\n\n"

        context += (
            f"## Instructions\n"
            f"Create exactly ONE new phase to advance this project.\n"
            f"Use phase ID 'phase-{next_phase_num}' and task IDs starting from "
            f"'task-{next_task_start:03d}'.\n"
            f"Respond with JSON only (no markdown):\n"
            f'{{"phases": [{{"id": "phase-{next_phase_num}", "name": "...", '
            f'"description": "...", "tasks": [{{"id": "task-{next_task_start:03d}", '
            f'"description": "..."}}], "entry_criteria": [...], "exit_criteria": [...]}}]}}'
        )

        planner = DefaultAgent(
            self._config,
            self._secrets,
            tools=[],
            name="orchestrator-advance-planner",
            model_selector=self._model_selector,
        )

        result = await planner.invoke(context, channel="workflow")

        # Parse the new phase
        plan_text = result.strip()
        if plan_text.startswith("```"):
            lines = plan_text.split("\n")
            plan_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            plan = json.loads(plan_text)
            phases_data = plan.get("phases", [])
            if not phases_data:
                raise ValueError("No phases in LLM response")

            pd = phases_data[0]  # Take only the first phase
            phase_id = pd.get("id", f"phase-{next_phase_num}")
            tasks = []
            for j, td in enumerate(pd.get("tasks", [])):
                task_id = td.get("id", f"task-{next_task_start + j:03d}")
                tasks.append(
                    TaskSpec(
                        id=task_id,
                        description=td.get("description", ""),
                        allowed_tools=td.get("allowed_tools", []),
                    )
                )

            entry_criteria = [PhaseEntry(description=ec) for ec in pd.get("entry_criteria", [])]
            exit_criteria = [PhaseEntry(description=ec) for ec in pd.get("exit_criteria", [])]

            new_phase = Phase(
                id=phase_id,
                name=pd.get("name", f"Phase {next_phase_num}"),
                description=pd.get("description", directive or "Continue project"),
                tasks=tasks,
                entry_criteria=entry_criteria,
                exit_criteria=exit_criteria,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse advance plan JSON: {e}")
            # Fallback: create a simple phase from the directive
            new_phase = Phase(
                id=f"phase-{next_phase_num}",
                name=directive[:50] if directive else "Continue",
                description=directive or "Continue making progress on the project goal",
                tasks=[TaskSpec(id=f"task-{next_task_start:03d}", description=directive or project.goal)],
            )

        project.phases.append(new_phase)
        project.journal.append(
            JournalEntry(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                actor="orchestrator",
                action="advanced",
                content=f"Added phase {new_phase.id}: {new_phase.name}",
            )
        )
        return new_phase

    # -- Internal: execution -------------------------------------------------

    async def _execute_phases(self, project: Project) -> None:
        """Execute phases sequentially (v1)."""
        for phase in project.phases:
            if project.status in {ProjectStatus.PAUSED, ProjectStatus.CANCELLED}:
                break

            if phase.status in {PhaseStatus.COMPLETED, PhaseStatus.SKIPPED}:
                continue

            # Check budget
            if project.budget.exhausted:
                project.journal.append(
                    JournalEntry(
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        actor="orchestrator",
                        action="budget_warning",
                        content=(
                            f"Budget exhausted: "
                            f"${project.budget.cost_used_usd:.2f}/${project.budget.max_cost_usd:.2f} USD, "
                            f"{project.budget.llm_calls_used}/{project.budget.max_llm_calls} calls"
                        ),
                    )
                )
                evt = self._backend.create_event(
                    project.id,
                    BUDGET_EXHAUSTED,
                    "orchestrator",
                    {"detail": f"cost={project.budget.cost_used_usd}, calls={project.budget.llm_calls_used}"},
                )
                await self._backend.record_event(project.id, evt)
                project.last_event_seq = evt.seq
                await self._transition(project, ProjectStatus.PAUSED)
                break

            # Budget warnings at thresholds
            remaining = project.budget.remaining_fraction()
            for threshold, pct in [(0.5, "50%"), (0.2, "80%"), (0.05, "95%")]:
                if remaining <= threshold:
                    evt = self._backend.create_event(
                        project.id,
                        BUDGET_WARNING,
                        "orchestrator",
                        {"threshold": pct},
                    )
                    await self._backend.record_event(project.id, evt)
                    project.last_event_seq = evt.seq
                    break

            # Start phase
            await self._start_phase(project, phase)

            # Execute tasks
            success = await self._execute_phase_tasks(project, phase)

            if success:
                await self._complete_phase(project, phase)
            else:
                if project.status not in {ProjectStatus.PAUSED, ProjectStatus.CANCELLED}:
                    await self._fail_phase(project, phase)
                break

        # Check terminal conditions after all phases
        if project.status == ProjectStatus.EXECUTING:
            all_done = all(p.status in {PhaseStatus.COMPLETED, PhaseStatus.SKIPPED} for p in project.phases)
            if all_done:
                await self._transition(project, ProjectStatus.REVIEWING)
                # Evaluate terminal conditions
                await self._evaluate_terminal_conditions(project)

    async def _start_phase(self, project: Project, phase: Phase) -> None:
        """Mark a phase as active."""
        phase.status = PhaseStatus.ACTIVE
        phase.started = datetime.now().isoformat(timespec="seconds")

        evt = self._backend.create_event(project.id, PHASE_STARTED, "orchestrator", {"phase_id": phase.id})
        await self._backend.record_event(project.id, evt)
        project.last_event_seq = evt.seq

        project.journal.append(
            JournalEntry(
                timestamp=phase.started,
                actor="orchestrator",
                action="phase_started",
                content=f"Started phase: {phase.name}",
            )
        )
        await self._report(f"Phase started: {phase.name}")
        await self._store.update(project)

    async def _execute_phase_tasks(self, project: Project, phase: Phase) -> bool:
        """Execute all tasks in a phase. Returns True if all succeeded."""
        if not self._worker_pool:
            project.journal.append(
                JournalEntry(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    actor="orchestrator",
                    action="error",
                    content="No worker pool available",
                )
            )
            return False

        all_success = True
        for task in phase.tasks:
            if project.status in {ProjectStatus.PAUSED, ProjectStatus.CANCELLED}:
                return False
            if project.budget.exhausted:
                return False

            result = await self._worker_pool.spawn_worker(project, phase, task)
            if not result.success:
                all_success = False
                # Check retry
                if task.max_attempts > 1:
                    for attempt in range(2, task.max_attempts + 1):
                        result = await self._worker_pool.spawn_worker(project, phase, task, attempt=attempt)
                        if result.success:
                            all_success = True
                            break
                if not result.success:
                    project.journal.append(
                        JournalEntry(
                            timestamp=datetime.now().isoformat(timespec="seconds"),
                            actor="orchestrator",
                            action="error",
                            content=f"Task {task.id} failed: {result.error}",
                        )
                    )
                    break

        return all_success

    async def _complete_phase(self, project: Project, phase: Phase) -> None:
        """Mark a phase as completed."""
        phase.status = PhaseStatus.COMPLETED
        phase.completed = datetime.now().isoformat(timespec="seconds")

        evt = self._backend.create_event(project.id, PHASE_COMPLETED, "orchestrator", {"phase_id": phase.id})
        await self._backend.record_event(project.id, evt)
        project.last_event_seq = evt.seq

        project.journal.append(
            JournalEntry(
                timestamp=phase.completed,
                actor="orchestrator",
                action="phase_completed",
                content=f"Completed phase: {phase.name}",
            )
        )
        await self._report(f"Phase completed: {phase.name}")
        await self._store.update(project)

    async def _fail_phase(self, project: Project, phase: Phase) -> None:
        """Mark a phase as failed."""
        phase.status = PhaseStatus.FAILED

        evt = self._backend.create_event(
            project.id,
            PHASE_FAILED,
            "orchestrator",
            {"phase_id": phase.id, "error": "Task(s) failed"},
        )
        await self._backend.record_event(project.id, evt)
        project.last_event_seq = evt.seq

        project.journal.append(
            JournalEntry(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                actor="orchestrator",
                action="phase_failed",
                content=f"Phase failed: {phase.name}",
            )
        )
        await self._transition(project, ProjectStatus.FAILED)
        await self._store.update(project)

    # -- Internal: terminal conditions ---------------------------------------

    async def _evaluate_terminal_conditions(self, project: Project) -> None:
        """Evaluate all terminal conditions. If all met, transition to DONE."""
        if not project.terminal_conditions:
            # No explicit conditions — all phases done means done
            await self._transition(project, ProjectStatus.DONE)
            await self._report(f"Project {project.id} completed!")
            await self._store.update(project)
            return

        all_met = True
        for tc in project.terminal_conditions:
            met = await self._evaluate_condition(project, tc)
            tc.met = met
            if met:
                tc.met_at = datetime.now().isoformat(timespec="seconds")

            evt = self._backend.create_event(
                project.id,
                TERMINAL_CONDITION_EVALUATED,
                "orchestrator",
                {"condition": tc.description, "type": tc.type, "met": met},
            )
            await self._backend.record_event(project.id, evt)
            project.last_event_seq = evt.seq

            if not met:
                all_met = False

        if all_met:
            await self._transition(project, ProjectStatus.DONE)
            await self._report(f"Project {project.id} completed! All terminal conditions met.")
        else:
            unmet = [tc.description for tc in project.terminal_conditions if not tc.met]
            await self._report(f"Project {project.id} reviewing. Unmet conditions: {', '.join(unmet)}")

        await self._store.update(project)

    async def _evaluate_condition(self, project: Project, tc: TerminalCondition) -> bool:
        """Evaluate a single terminal condition."""
        if tc.type == "artifact_exists":
            artifact_name = tc.params.get("name", "")
            return any(a.name == artifact_name for a in project.artifacts)

        elif tc.type == "phase_count":
            required = tc.params.get("min_completed", len(project.phases))
            completed = sum(1 for p in project.phases if p.status == PhaseStatus.COMPLETED)
            return completed >= required

        elif tc.type == "llm_eval":
            # Use LLM to evaluate the condition
            try:
                from roshni.agent.default import DefaultAgent

                evaluator = DefaultAgent(
                    self._config,
                    self._secrets,
                    tools=[],
                    name="condition-evaluator",
                    model_selector=self._model_selector,
                )
                prompt = (
                    f"Evaluate whether this condition is met for the project.\n\n"
                    f"Project goal: {project.goal}\n"
                    f"Condition: {tc.description}\n"
                    f"Phases completed: "
                    f"{sum(1 for p in project.phases if p.status == PhaseStatus.COMPLETED)}"
                    f"/{len(project.phases)}\n"
                    f"Artifacts: {', '.join(a.name for a in project.artifacts)}\n\n"
                    f'Respond with JSON: {{"met": true/false, "rationale": "...", "evidence": [...]}}'
                )
                result = await evaluator.invoke(prompt, channel="workflow")
                # Parse response
                result_text = result.strip()
                if result_text.startswith("```"):
                    lines = result_text.split("\n")
                    result_text = "\n".join(lines[1:-1])
                eval_data = json.loads(result_text)
                tc.evaluation = eval_data
                return eval_data.get("met", False)
            except Exception as e:
                logger.warning(f"LLM eval failed for condition: {e}")
                return False

        elif tc.type == "check_fn":
            # Custom check function — not implemented in v1
            return False

        return False

    # -- Internal: transitions -----------------------------------------------

    async def _transition(self, project: Project, target: ProjectStatus) -> None:
        """Transition project status with validation and event recording."""
        old_status = project.status
        validate_transition(old_status, target)
        project.status = target
        project.updated = datetime.now().isoformat(timespec="seconds")

        if target == ProjectStatus.EXECUTING and not project.started_at:
            project.started_at = project.updated
        if target == ProjectStatus.CANCELLED:
            project.cancel_requested_at = project.updated

        evt = self._backend.create_event(
            project.id,
            PROJECT_TRANSITIONED,
            "orchestrator",
            {"from": old_status.value, "to": target.value},
        )
        await self._backend.record_event(project.id, evt)
        project.last_event_seq = evt.seq

        project.journal.append(
            JournalEntry(
                timestamp=project.updated,
                actor="orchestrator",
                action="status_change",
                content=f"{old_status.value} -> {target.value}",
            )
        )

    # -- Internal: reporting -------------------------------------------------

    async def _report(self, message: str) -> None:
        """Send a status update to the user via send_fn."""
        if self._send_fn:
            try:
                import inspect

                if inspect.iscoroutinefunction(self._send_fn):
                    await self._send_fn(message)
                else:
                    self._send_fn(message)
            except Exception as e:
                logger.warning(f"Failed to report to user: {e}")
