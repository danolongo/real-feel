# Writing Rules

These rules apply to every document, code comment, commit message, and PR description in Real Feel.

Three rules:

1. Every word has exactly one definition and one part of speech. Real Feel bans ambiguity.
2. Always write instructions so it is clear who or what is performing the action. Avoid passive phrasing that hides the actor.
3. Keep sentences short and punchy.

## Rule 1: One word, one meaning

[glossary.md](glossary.md) defines every Real Feel term.

- Use the glossary term. Do not swap in a synonym for variety.
- Use each term as one part of speech only.
- Add a new term to the glossary in the same PR that first uses it.
- Name code after glossary terms: types, functions, tables, queues, environment variables.
- Repeat the noun when a pronoun could point to two things.
- Outside the glossary, pick one sense for a common word and keep that sense in the whole document.

<!-- vale off -->

| Bad | Good | Why |
|-----|------|-----|
| The crawler sends the request to Kafka. | The scraper sends the job to the `classifier-jobs` queue. | "crawler", "request" and "Kafka" are not Real Feel terms. |
| Label each tweet. | Attach a label to each tweet. | "label" is a noun only. |
| Batch the inputs. | Group the model inputs into batches. | "batch" is a noun only. |
| The model returns a score. | Clef-flash returns a bot probability. | "score" is banned. "the model" is ambiguous. |
| It retries until it fails. | The worker retries the job until the job reaches the DLQ. | "it" pointed to two things. |

<!-- vale on -->

## Rule 2: Name the actor

Make the actor the subject of the sentence. The reader must always know who or what acts.

- In instructions, the actor is "you".
- In system descriptions, the actor is a component from the glossary: the API, the worker, the scraper, the classifier, CI, Postgres, SQS.
<!-- vale off -->

- Avoid "is done", "gets sent", "will be stored". Name who does it.

| Bad | Good |
|-----|------|
| The job is deleted after success. | The worker deletes the job after Postgres commits the predictions. |
| Migrations should be run first. | You run `cargo sqlx migrate run` first. |
| Tests are triggered on push. | CI runs the tests on every push to `main`. |
| The status gets updated. | The classifier worker sets the query status to `done`. |

<!-- vale on -->

## Rule 3: Short sentences

- Write one idea per sentence.
- Aim for 15 words. Never pass 25.
- Prefer a list over a long paragraph.
- Cut filler: "basically", "simply", "just", "actually", "really", "in order to", "note that".
- Put the main point first.

| Bad | Good |
|-----|------|
<!-- vale off -->

| In order to make sure that the worker doesn't end up processing the same job twice, which could happen because SQS delivers at least once, we basically use upserts. | SQS can deliver a job twice. The worker writes with upserts. A second delivery changes nothing. |

<!-- vale on -->

## Commit messages

Every commit uses this format:

```text
<type>(<regarding>): <brief and clear description>

<One paragraph that explains the changes.>
```

- **`<type>`** names the kind of change: `add`, `fix`, `refactor`, `docs`, `test`, `infra`, `chore`.
- **`<regarding>`** names the part of Real Feel the change touches: a crate name (`api`, `db`, `worker`), `infra`, `ci`, or `docs`.
- **The description** says what the commit does, in lower case, with no full stop. Keep it under 72 characters.
- **The body** is one paragraph. It explains what changed and why. It never runs to a second paragraph.
- **No co-authors.** No commit carries a `Co-authored-by` line, for a person or a tool.
- **The writing rules apply.** Name the actor, use glossary terms, and keep sentences short.

| Bad | Good |
|-----|------|
| `updated stuff` | `fix(worker): delete the job only after Postgres commits` |
| `feat: Add Migrations.` | `add(db): migrations for queries and tweets` |
| A body with three paragraphs and a `Co-authored-by` line | One paragraph, no trailer |

Example:

```text
fix(worker): delete the job only after Postgres commits

The classifier worker deleted the classifier job before the predictions
transaction committed. A crash between the two steps lost the batch. The
worker now commits first and deletes the job second, so SQS redelivers
the job after a crash and the upserts make the retry safe.
```

## Module docs

Every module has a `README.md`. Write it as an onboarding document for a developer on their first day.

Use this template. Drop a section when the module has nothing to put in it.

```markdown
# <module name>

<One sentence: what this module does.>

## Where it sits
<A small diagram or two sentences: what calls this module, and what this module calls.>

## Terms
<Glossary terms this module uses. Link to glossary.md. Do not redefine them.>

## How it works
<Numbered steps. Each step names its actor.>

## Configuration
| Variable | Default | Meaning |
|----------|---------|---------|

## Run it
<Exact commands. State the working directory.>

## Test it
<Exact commands. List which tests need Docker or network access.>

## Failure modes
<What breaks, what the module does when it breaks, and what you do.>

## Decisions
<Names of the decision records that shaped this module. Do not link them: git ignores `docs/decisions/`.>
```

## Decision records

You write one decision record for each architecture decision. Store it as `docs/decisions/NNNN-short-title.md`. Git ignores that folder, so the records stay on your machine.

```markdown
# NNNN: <title>

Status: proposed | accepted | replaced by NNNN
Date: YYYY-MM-DD

## Context
<The problem. The facts that matter.>

## Decision
<What Real Feel does. One paragraph.>

## Reasons
<A list.>

## Costs
<What this decision makes harder.>
```

## Enforcement

- The PR template lists all three rules as checkboxes. The reviewer checks them.
- CI runs Vale on every Markdown file. Vale flags passive voice, long sentences, and words from the "Do not use" column of the glossary.
- When the glossary changes, you update the Vale substitution list in the same PR.
