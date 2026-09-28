# Loop Skills — Backlog Task Execution Workflow

Playbook for working through monday.com "Tasks" backlog items (board `Agent
Execution Guardian` MVP team backlog, board id `5031356734`) one by one.
Read this before starting a batch of backlog tasks.

## Sequence, per task

1. **Read the task.** Treat its Acceptance Criteria as the objective —
   don't drift from what it actually asks for.
2. **List, before writing any code:**
   1. Advantages of having this task.
   2. How it impacts the current flow.
   3. Extra steps it adds and the impact to latency.
   4. Files that will be created or changed.
   5. Hardcoded values this change introduces.
   6. All test cases, including edge cases covered.
3. **Update the monday.com backlog item** (item update / comment) with the
   above before writing code.
4. **Architecture decisions with two viable options:** choose one using
   judgment and record the rationale in the same backlog item update. Don't
   leave the decision implicit in code — write down *why*.
5. **Write tests first (TDD).** Tests encode the list from step 2.6.
   Confirm they fail for the right reason (red) before implementing.
6. **Close the gap** — implement until the tests from step 5 pass (green).
7. **Update the backlog item again** listing exactly which files were
   created/changed to close the task.

## Standing rules (apply to every task, not just the sequence above)

- **Do not delete files without confirmation.** Ask first, always — this
  includes files created earlier in the same task, not just pre-existing
  ones.
- **Stay inside the project folder.** Don't read, write, or execute
  anything outside `agent-circuit-breaker/` — this includes not using an
  external scratch/temp directory for exploration. If a throwaway probe is
  needed (e.g. empirically confirming a library's behavior before writing
  an assertion), prefer an inline command that writes nothing to disk
  (e.g. `python3 -c "..."`) over creating a file; if a file genuinely must
  exist to run something, put it inside the project and it still can't be
  deleted without confirmation per the rule above.
- **Do not push to any GitHub remote** until explicitly asked, separately
  from being asked to do the work itself.
- **On the "apex" skill:** `.agents/skills/apex/` is authored for a
  different agent runtime (`disable-model-invocation: true`,
  `opencode/autoinvoke: false`, ships an `agents/openai.yaml` vendor
  config) and is not in Claude Code's invokable skill list. Don't try to
  invoke it as a packaged skill — follow the numbered sequence above
  directly instead. (Confirmed with the user 2026-09-21.)
- **Keep progressing to tasks in the same phase without user intervention.**
  Once told to work a phase (e.g. "work on Phase 1"), that's authorization
  for every task in that phase, not just the first one. On finishing one
  task (tests green, backlog item updated, marked Done), move straight to
  the next task in the same phase — don't pause to ask "should I continue?"
  or "want me to keep going?" between tasks. Still run the full per-task
  sequence above for each one (analysis → monday.com update → architecture
  decision + rationale → tests → implementation → final monday.com update
  → mark Done), and still give a brief status message as each task
  finishes — this rule removes the *stop-and-wait-for-a-go-ahead* between
  tasks, not the communication.

  Stop and check in with the user instead of auto-continuing when:
  - Every task in the current phase is Done or otherwise unactionable —
    confirm before moving on to the next phase; phase boundaries are still
    a checkpoint.
  - A task is genuinely blocked (e.g. it depends on a task from a
    different, not-yet-built phase, or on an unresolved decision tracked
    elsewhere on the board).
  - Another standing rule in this file already calls for confirmation
    (deleting a file, working outside the project folder, pushing to
    GitHub) — those exceptions are never overridden by this one.

## Notes

- This file is the authoritative process reference for backlog-task
  execution — check here before assuming the workflow from a single past
  conversation still applies.
- Update this file itself when the user changes or adds to the rules,
  rather than letting the rules live only in chat history.
