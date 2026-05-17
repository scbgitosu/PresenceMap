# AGENTS.md instructions for PresenceMap

PresenceMap is a local fork of HeatMap for RF-based motion and presence
detection experiments.

## Current Repository State

- This repo lives at `/Users/scottbrough/Projects/PresenceMap`.
- It was cloned from `/Users/scottbrough/Projects/HeatMap`.
- The inherited HeatMap remote is named `heatmap-upstream`.
- The project is not yet pushed to GitHub.
- GitNexus may still know the original project as `HeatMap`; use that index only
  for inherited code understanding until PresenceMap is indexed separately.

## Working Rules

- Keep HeatMap and PresenceMap changes separate.
- Do not push to the inherited HeatMap remote.
- Prefer adding PresenceMap functionality beside inherited survey code before
  deleting or renaming large modules.
- Before editing inherited Python functions, classes, or methods, use the
  HeatMap GitNexus index for impact analysis when available.
- Preserve the HP Linux collector plus Mac Streamlit analysis split unless the
  user explicitly asks for a different architecture.

## Product Direction

PresenceMap should focus on:

- continuous RF observation
- motion tripwire detection
- room-level presence estimation
- calibration workflows
- local event logs
- optional automation outputs such as MQTT or Home Assistant webhooks

PresenceMap should not claim precise person identification from ordinary Wi-Fi
scan data alone. Treat identity detection as future research that likely needs
additional hardware or signal sources.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **PresenceMap** (1673 symbols, 2693 relationships, 95 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/PresenceMap/context` | Codebase overview, check index freshness |
| `gitnexus://repo/PresenceMap/clusters` | All functional areas |
| `gitnexus://repo/PresenceMap/processes` | All execution flows |
| `gitnexus://repo/PresenceMap/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
