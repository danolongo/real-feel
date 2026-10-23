# Real Feel: Claude context

Real Feel finds bots on X. A user sends a query. The scraper fetches tweets through Firecrawl. Clef classifies the account behind each tweet. The API returns the result. There is no X API.

The system is an all-Rust workspace on AWS. `docs/refactor-plan.md` holds the plan, the milestones, and the gates. M0 finished. The next milestone is M1.

## Rules that override defaults

- Follow `docs/writing-rules.md` in every doc, code comment, commit, and PR. Use the terms in `docs/glossary.md`.
- Never add a `Co-authored-by` line or any other attribution trailer to a commit.
- Work on one branch and one PR for each task.
- Add a new term to the glossary in the same PR that first uses it. Update the Vale rules in `.vale/styles/RealFeel/` too.
- `docs/decisions/` is local. Git ignores it. Never commit or link it.

## Commands

`docs/local-dev.md` has the full setup and the CI checks.

## Dev preferences

- The user writes first. Claude reviews and fixes.
- Implement exactly what the task needs. No extra abstractions.
- Write no comment unless the reason is not obvious from the code.
- Keep explanations short. Skip filler.
- Prefer pure Rust crates over crates that bind C libraries.
- Use `uv` for Python dependencies.
