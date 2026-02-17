# Paper Review: Intelligent AI Delegation

**Paper:** [Intelligent AI Delegation](https://arxiv.org/abs/2602.11865)
**Authors:** Nenad Tomašev, Matija Franklin, Simon Osindero (Google DeepMind)
**Published:** 2026-02-12

## Summary

The paper proposes a framework for intelligent AI delegation — structured task allocation
that incorporates transfer of authority, responsibility, accountability, role clarity, and
trust mechanisms. It targets multi-agent systems where agents delegate tasks to other agents
(or humans) in complex chains.

Five core pillars: **Dynamic Assessment**, **Adaptive Execution**, **Structural Transparency**,
**Scalable Market Coordination**, and **Systemic Resilience**.

Key concepts:
- **Contract-first decomposition** — tasks decomposed until sub-units have automated verification
- **Delegation Capability Tokens (DCTs)** — scoped, attenuated authorization tokens
- **Transitive accountability** — in chain A→B→C, B is accountable for verifying C
- **Trust & reputation** — dynamic trust scores + immutable performance histories
- **Circuit breakers** — auto-revoke delegation authority when trust drops
- **Zero-trust philosophy** — every delegation requires constraints + verification

---

## Ideas for Roshni

### 1. Delegation Capability Tokens (DCTs)

**Paper concept:** Agents receive scoped, time-limited tokens (inspired by Macaroons/Biscuits)
with cryptographic caveats enforcing least privilege. A token might allow READ on a specific
folder but forbid WRITE.

**Current state:** `PermissionTier` is a static integer enum (NONE/OBSERVE/INTERACT/FULL).
`LayeredToolPolicy` composes allowlist/blocklist at config time. Neither is contextual or
attenuable at runtime.

**Idea:** Introduce a `DelegationToken` that carries:
- Granted permissions (subset of delegator's permissions)
- Scope constraints (specific tools, specific resource paths)
- Time-to-live
- Delegation depth limit (how many times it can be re-delegated)
- Issuer chain (who granted what)

When the Router dispatches to an agent, or when a future multi-agent system has agents calling
other agents, the callee receives a token rather than inheriting the caller's full permissions.
This prevents privilege escalation in delegation chains.

**Where it fits:**
- `agent/permissions.py` — extend beyond static tiers
- `agent/tool_policy.py` — runtime token checking instead of config-only filtering
- New `agent/delegation_token.py` — token issuance, validation, attenuation

---

### 2. Contract-First Task Decomposition

**Paper concept:** Tasks should only be delegated if the outcome can be precisely verified.
If a task is too subjective, recursively decompose it until sub-tasks match available
verification tools (unit tests, formal proofs, output schemas).

**Current state:** `TaskStore` supports dependencies (`depends_on[]`) and status transitions,
but tasks have no verification criteria. Completion is a manual status change — there's no
mechanism to programmatically verify that a task was actually done correctly.

**Idea:** Add verification contracts to tasks:
```python
@dataclass
class VerificationContract:
    """What constitutes successful task completion."""
    output_schema: dict | None = None       # JSON Schema for expected output
    verification_fn: str | None = None      # Callable path for programmatic check
    acceptance_criteria: list[str] = field(default_factory=list)  # Human-readable criteria
    verifier: str | None = None             # Agent/human who verifies
```

Integrate into the task lifecycle:
- Task creation requires a contract (even if it's just acceptance criteria)
- `TaskStore.complete_task()` runs verification before allowing DONE transition
- Unverifiable tasks must be decomposed further before delegation

**Where it fits:**
- `agent/task_store.py` — add `VerificationContract` to `Task`
- `agent/default.py` — tool loop checks verification before marking tasks done

---

### 3. Transitive Accountability Chain

**Paper concept:** In delegation chain A→B→C, agent B is responsible for C's work. Agent A
performs a 2-stage check: verify B's direct work, then verify B correctly verified C. Agents
cannot absolve themselves by blaming sub-contractors.

**Current state:** The Router dispatches to agents, but there's no record of delegation chains.
The event bus emits `AGENT_TOOL_CALLED` and `AGENT_TOOL_RESULT`, but these don't capture
who-delegated-to-whom or verification outcomes. The audit log in VaultManager is append-only
text with no structured accountability.

**Idea:** Introduce a `DelegationChain` that tracks:
```python
@dataclass
class DelegationRecord:
    delegator: str              # Agent/user who delegated
    delegatee: str              # Agent who received the task
    task_id: str                # What was delegated
    token: DelegationToken      # Authorization granted
    timestamp: datetime
    verification_result: VerificationResult | None = None
    sub_delegations: list[DelegationRecord] = field(default_factory=list)
```

This creates an auditable tree of who did what and whether each step was verified. The
delegator can inspect the full chain before accepting results.

**Where it fits:**
- New `agent/delegation.py` — DelegationRecord, DelegationChain
- `agent/default.py` — record delegations in tool loop
- `agent/vault.py` — persist chains alongside audit log

---

### 4. Trust & Reputation System

**Paper concept:** Trust is a dynamic, contextual belief about a delegatee's capability.
Reputation is an immutable performance history. Trust is updated based on verifiable data
streams (task completion rates, verification outcomes, latency).

**Current state:** `CircuitBreaker` tracks binary pass/fail per service with a rolling window.
It auto-trips after N consecutive failures and resets after a cooldown. This is a blunt
instrument — no gradation, no per-agent tracking, no reputation history.

**Idea:** Extend CircuitBreaker into a `TrustManager`:
```python
class TrustManager:
    """Tracks trust scores and reputation for agents and services."""

    def record_outcome(self, entity: str, task_type: str, outcome: Outcome) -> None:
        """Record task outcome — updates trust score and reputation."""

    def get_trust(self, entity: str, task_type: str | None = None) -> float:
        """Current trust score (0.0-1.0), optionally per task type."""

    def get_reputation(self, entity: str) -> ReputationReport:
        """Immutable performance history summary."""

    def is_trusted_for(self, entity: str, task_type: str, min_trust: float) -> bool:
        """Can this entity be trusted for this task type?"""
```

Trust scores inform delegation decisions:
- High trust → delegate with less oversight
- Low trust → require verification or refuse delegation
- Sudden drops → trigger circuit breaker (auto-revoke active tokens)

**Where it fits:**
- `agent/circuit_breaker.py` — evolve into or compose with TrustManager
- `agent/default.py` — record outcomes after tool calls
- `agent/router.py` — consider trust when selecting agents

---

### 5. Graduated Human Oversight

**Paper concept:** Preserve meaningful human oversight through "cognitive friction" — explicit
transfer points where authority shifts, and curriculum-aware routing that prevents de-skilling.
Oversight scales with task risk.

**Current state:** The approval workflow is binary: write tools either require approval or
don't. `auto_approve_channels` bypasses approval entirely. There's no risk assessment.

**Idea:** Replace binary approval with graduated oversight based on impact assessment:

| Risk Level | Oversight | Example |
|-----------|-----------|---------|
| LOW | Auto-approve, log | Read operations, search |
| MEDIUM | Notify, auto-approve after delay | Send a message to known contact |
| HIGH | Require explicit approval | Financial transaction, delete data |
| CRITICAL | Require approval + confirmation | Bulk operations, external API writes |

Add an `ImpactAssessor` that evaluates:
- Tool permission level
- Scope of effect (single record vs. bulk)
- Reversibility
- Historical patterns (first time using this tool? unusual parameters?)

**Where it fits:**
- `agent/approval.py` — extend beyond binary approve/deny
- New `agent/impact.py` — ImpactAssessor with risk scoring
- `agent/default.py` — integrate assessment into tool execution path

---

### 6. Verification Protocol for Tool Results

**Paper concept:** Three verification mechanisms: (1) direct outcome inspection for
high-verifiability tasks, (2) trusted third-party auditing, (3) cryptographic proofs.

**Current state:** Tool results are passed directly back to the LLM with no verification.
`ToolDefinition.execute()` has retry logic for transient errors but no output validation.

**Idea:** Add a `Verifier` protocol:
```python
@runtime_checkable
class Verifier(Protocol):
    def verify(self, tool_name: str, arguments: dict, result: Any) -> VerificationResult:
        """Verify a tool result. Returns pass/fail with explanation."""
```

Built-in verifiers:
- **SchemaVerifier** — validates result against JSON Schema
- **BoundsVerifier** — checks numeric results are within expected ranges
- **IdempotencyVerifier** — for read tools, checks result stability across calls
- **LLMVerifier** — uses a separate (cheaper) model to sanity-check results

**Where it fits:**
- New `agent/verification.py` — Verifier protocol + built-in implementations
- `agent/tools/__init__.py` — attach verifiers to ToolDefinitions
- `agent/default.py` — run verification in tool loop before returning result to LLM

---

### 7. Dynamic Permission Attenuation

**Paper concept:** The "confused deputy problem" — an agent with broad permissions is tricked
into misusing them. DCTs ensure each delegation narrows permissions.

**Current state:** Tool policies are layered (global → channel → agent) but all configured
at startup. An agent's permissions don't narrow based on what specific task it's performing.

**Idea:** Make permissions context-sensitive:
- When processing a user query about weather, the agent's active permission set should
  temporarily narrow to only weather-related tools
- Router could infer task category from the query and apply a transient policy overlay
- ModelSelector already classifies query complexity — extend this to classify permission needs

This is lighter-weight than full DCTs but captures the core benefit: agents operate with
minimum necessary permissions for the task at hand.

**Where it fits:**
- `agent/tool_policy.py` — add `TransientPolicy` layer
- `agent/router.py` — infer task category, apply policy overlay
- `agent/default.py` — activate transient policy per chat turn

---

### 8. Structured Delegation Events

**Paper concept:** Delegation is a first-class operation with clear lifecycle events:
initiate, accept, execute, verify, complete/escalate.

**Current state:** The event bus emits generic events (CHAT_START, TOOL_CALLED, etc.).
There's no concept of delegation as a distinct event type.

**Idea:** Add delegation-specific events:
```python
class DelegationEvent(Enum):
    DELEGATION_INITIATED = "delegation.initiated"
    DELEGATION_ACCEPTED = "delegation.accepted"
    DELEGATION_REJECTED = "delegation.rejected"
    DELEGATION_COMPLETED = "delegation.completed"
    DELEGATION_FAILED = "delegation.failed"
    DELEGATION_ESCALATED = "delegation.escalated"
    DELEGATION_REVOKED = "delegation.revoked"
```

This enables monitoring, alerting, and analytics on delegation patterns. Combined with the
TrustManager, the system could detect and flag anomalous delegation chains.

**Where it fits:**
- `agent/default.py` — emit delegation events
- Event consumers (logging, metrics, alerting)

---

## Priority Assessment

| Idea | Impact | Effort | Priority |
|------|--------|--------|----------|
| 4. Trust & Reputation | High | Medium | **P1** — Extends existing CircuitBreaker naturally |
| 2. Contract-First Tasks | High | Medium | **P1** — Adds rigor to existing TaskStore |
| 5. Graduated Oversight | High | Low | **P1** — Improves existing approval workflow |
| 6. Verification Protocol | High | Medium | **P2** — Foundational for other ideas |
| 7. Dynamic Attenuation | Medium | Low | **P2** — Quick win on existing policy system |
| 1. Delegation Tokens | High | High | **P3** — Most impactful but needs multi-agent first |
| 3. Accountability Chain | Medium | Medium | **P3** — Depends on delegation tokens |
| 8. Delegation Events | Low | Low | **P3** — Nice-to-have, depends on multi-agent |

**Recommended starting point:** Ideas 4 (Trust), 2 (Contracts), and 5 (Graduated Oversight)
build directly on existing infrastructure and deliver value even in the current single-agent
architecture. They also lay the foundation for ideas 1, 3, and 6 when multi-agent delegation
becomes a priority.
