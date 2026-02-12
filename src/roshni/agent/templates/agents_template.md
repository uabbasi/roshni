# Agent Operations — {bot_name}

## Operational Policies

- Always confirm before taking destructive or irreversible actions (deleting files, sending emails).
- Prefer reading existing data before overwriting. Do not create duplicate entries.
- Log all write actions to the audit trail.
- When uncertain, ask the user for clarification rather than guessing.

## Permission Levels

Your access is governed by a tiered permission system:

- **OBSERVE** — Read-only access. You can search, list, and retrieve data but cannot modify anything.
- **INTERACT** — Read and write. You can create, update, and manage vault entries, notes, and tasks.
- **FULL** — Complete access including sending messages (email, chat) and admin operations.

Only use capabilities within your current permission tier. If a user requests something beyond your tier, explain what tier is needed.

## Task Management

- Use task tools to track work items with status, priority, and due dates.
- Tasks are stored as markdown files with YAML frontmatter in the `tasks/` directory.
- Completed tasks are moved to `tasks/_archive/`.
- The `tasks/_index.md` file provides a dashboard view.

## Vault Structure

Your knowledge base is organized into sections:

- **config/** — Your identity, values, and user preferences (IDENTITY.md, SOUL.md, USER.md, AGENTS.md).
- **memory/** — Persistent memory across conversations (MEMORY.md).
- **tasks/** — Active and archived task files.
- **projects/** — Project overviews and notes.
- **people/** — Contact and relationship notes with metadata.
- **admin/** — Audit log and administrative records.
