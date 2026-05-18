# PresenceMap Agent Notes

PresenceMap is a local fork of HeatMap for RF-based motion, presence, and
automation experiments.

The inherited HeatMap codebase can still be used as the reference architecture:

- `hp_collector/` contains the HP Linux-side field collection patterns.
- `mac_analysis/` contains Mac-side Streamlit dashboards and floorplan tooling.
- `shared/` contains common models, CSV schemas, project paths, and metrics.

The immediate direction is to add a continuous collection path and event-scoring
pipeline while preserving the original survey workflow as useful reference code.

Before editing inherited functions, classes, or methods, consult GitNexus impact
analysis for the original HeatMap symbols when available. PresenceMap should be
indexed separately once its first code changes are in place.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **PresenceMap** (2700 symbols, 4352 relationships, 98 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
