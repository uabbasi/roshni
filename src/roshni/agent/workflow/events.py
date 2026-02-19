"""Canonical event type constants for the workflow system.

Used in WorkflowEvent.type and for EventBus subscriptions.
These are the minimum event set for lossless state reconstruction
via replay.
"""

# --- Project lifecycle ---
PROJECT_CREATED = "project.created"
PROJECT_TRANSITIONED = "project.transitioned"  # payload: {from, to}
PROJECT_STEERED = "project.steered"  # payload: {direction}
PROJECT_ADVANCED = "project.advanced"  # payload: {directive, new_phase_id}

# --- Plan ---
PLAN_WRITTEN = "plan.written"  # payload: {plan_hash}

# --- Phase lifecycle ---
PHASE_STARTED = "phase.started"  # payload: {phase_id}
PHASE_COMPLETED = "phase.completed"  # payload: {phase_id}
PHASE_FAILED = "phase.failed"  # payload: {phase_id, error}

# --- Task lifecycle ---
TASK_DISPATCHED = "task.dispatched"  # payload: {phase_id, task_id, worker_id}
TASK_COMPLETED = "task.completed"  # payload: {phase_id, task_id, worker_id, attempt, ...}
TASK_FAILED = "task.failed"  # payload: {phase_id, task_id, worker_id, attempt, error, retryable}

# --- Budget ---
BUDGET_RECORDED_CALL = "budget.recorded_call"  # payload: {cost_usd, total_cost, total_calls}
BUDGET_WARNING = "budget.warning"  # payload: {threshold}  -- 50%, 80%, 95%
BUDGET_EXHAUSTED = "budget.exhausted"  # payload: {detail}

# --- Conflict ---
CONFLICT_DETECTED = "conflict.detected"  # payload: {reason}
CONFLICT_RECONCILED = "conflict.reconciled"  # payload: {strategy}

# --- Terminal conditions ---
TERMINAL_CONDITION_EVALUATED = "terminal_condition.evaluated"  # payload: {condition, result}

# All event types for validation
ALL_EVENT_TYPES = frozenset(
    {
        PROJECT_CREATED,
        PROJECT_TRANSITIONED,
        PROJECT_ADVANCED,
        PROJECT_STEERED,
        PLAN_WRITTEN,
        PHASE_STARTED,
        PHASE_COMPLETED,
        PHASE_FAILED,
        TASK_DISPATCHED,
        TASK_COMPLETED,
        TASK_FAILED,
        BUDGET_RECORDED_CALL,
        BUDGET_WARNING,
        BUDGET_EXHAUSTED,
        CONFLICT_DETECTED,
        CONFLICT_RECONCILED,
        TERMINAL_CONDITION_EVALUATED,
    }
)
