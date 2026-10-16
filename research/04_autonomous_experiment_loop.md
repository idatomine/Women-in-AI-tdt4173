# 04 – Running Claude Code autonomously for 12–24 h on the unit-commitment task

Researched 2026-10-15. Covers (1) the Claude Code features that matter for an unattended run, (2) what agentic ML research systems (AIDE, MLE-bench, MLAgentBench, AIRA, Agent K, AI Scientist, Karpathy's autoresearch) learned, and (3) a concrete scaffold for this project: predict 14 units x 168 hourly on/off probabilities per `Run No`, scored by micro ROC-AUC (`kaggle_metric.py`), ~2900 training runs.

Every claim below links to its source. Where a source was a secondary summary rather than the primary paper or doc, it is marked as such.

---

## TL;DR

1. **The harness decides the outcome more than the model does.** Every system that works has a frozen evaluator the agent cannot edit, one scalar metric, a fixed per-experiment time budget, an append-only experiment log, and keep/discard via git.
2. **Pick the driver deliberately.** For a session on a laptop that stays open, use `/loop` (self-paced) or `/goal`, with a project `.claude/loop.md` holding the protocol. For something sturdier, run a shell `while` loop around `claude -p` so each iteration starts with a fresh context and all state lives on disk.
3. **No prompts, no stalls.** Use auto mode or `acceptEdits` with an explicit `permissions.allow` list, plus `deny` rules and a `PreToolUse` hook that block edits to `evaluate.py`, the folds and the data. Auto mode falls back to prompting after 3 blocks in a row or 20 in total, and that alone can freeze an overnight run.
4. **Plan around usage limits.** Max has a 5-hour session window and a weekly cap. A 24 h run will hit the 5-hour window. Interactive sessions wait and continue on their own once the limit resets. Spend tokens on thinking, not on reading logs.
5. **Validation overfitting is the main ML risk.** Meta's AIRA study found that true test performance plateaus or falls as search goes on, and that the val/test gap costs 9–13 percentage points of medal rate. Use time-blocked CV folds with a gap (the runs are daily and their 168 h windows overlap), keep a locked holdout that the loop never sees, and limit Kaggle submissions to a few top-k candidates picked by humans.

---

## 1. Claude Code features relevant to a long unattended run

### 1.1 Ways to keep Claude working

| Mechanism | What it is | Fit for this project |
|---|---|---|
| **`/loop`** (bundled skill) | Re-runs a prompt in the open session. `/loop 30m <prompt>` runs on a fixed cron. `/loop <prompt>` is self-paced: Claude picks a 1 min–1 h delay after each iteration. A bare `/loop` runs `.claude/loop.md` if that file exists. Loops expire after 7 days. A session holds at most 50 tasks. Tasks fire only while Claude Code is running **and idle**, missed fires are not caught up, and a self-paced loop is **not restored on `--resume`**. [docs](https://code.claude.com/docs/en/scheduled-tasks) | Good. Put the full protocol in `.claude/loop.md`, which is re-read every iteration, so you can edit it mid-run. Its content is truncated after 25,000 bytes. |
| **`/goal <condition>`** | After every turn, a small model checks whether the condition holds. If not, Claude starts another turn. Add a bound such as "or stop after N turns". Goals survive `--resume`. A goal **pauses** on a usage limit and resumes when the session continues after the reset. It also works with `claude -p "/goal …"`. [docs](https://code.claude.com/docs/en/goal) | Good for "keep improving until 08:00 or until CV AUC ≥ X". The evaluator reads only the transcript, so the condition has to be something Claude prints, such as the leaderboard top line. |
| **Headless `claude -p`** | Non-interactive runs. Supports `--continue` / `--resume <id>`, `--output-format json` (includes `session_id` and cost), `--permission-mode`, `--allowedTools`, `--permission-prompts none`, and `--max-budget-usd`. `--bare` skips CLAUDE.md, hooks and skills, and **does not use the subscription login** (it needs `ANTHROPIC_API_KEY`), so do **not** use `--bare` on Max. [docs](https://code.claude.com/docs/en/headless) | The sturdiest option. A bash `while` loop calls `claude -p` once per experiment, so each call gets a fresh context and nothing depends on compaction. |
| **Desktop scheduled tasks** | Start a **fresh session** on a schedule (minimum 1 min) with local file access. They run only while the app is open and the machine is awake, and a sleeping machine skips runs. Each task has its own permission mode. A run in Manual mode stalls until someone approves the prompt, so click "Run now" once and choose "always allow". [docs](https://code.claude.com/docs/en/desktop-scheduled-tasks) | OK. Each run is an independent session that reads state from disk. |
| **Cloud routines (`/schedule`)** | Run in the cloud with a fresh GitHub clone and no local files. The minimum interval is 1 h. They count against the same subscription usage. It is a research preview. [docs](https://code.claude.com/docs/en/routines) | Poor fit. Training needs the local data, and that data should not be pushed to GitHub. |

**Practical constraints:** stop the Mac from sleeping, for example with `caffeinate -dimsu` in another terminal (a standard macOS utility). A closed lid still sleeps the machine. Scheduled `/loop` prompts fire only between turns, so a turn blocked on a 40-minute training job delays the next iteration. That is fine as long as experiments are time-boxed (section 2).

### 1.2 Permissions: avoiding prompts without giving the agent free rein

Modes ([permissions](https://code.claude.com/docs/en/permissions), [permission modes](https://code.claude.com/docs/en/permission-modes)):
- `acceptEdits`: file edits and `mkdir`/`mv`/`cp` run without asking. Any other shell command needs a `permissions.allow` rule.
- `auto`: a classifier model reviews actions instead of you. This is the default starting mode in recent versions. **Caveat for unattended runs:** "if the classifier blocks an action 3 times in a row or 20 times total, auto mode pauses and Claude Code resumes prompting". Those thresholds cannot be configured. Explicit allow rules for every routine command keep the classifier out of the loop.
- `dontAsk`: anything not pre-approved is denied rather than prompted. This suits CI and `claude -p`.
- `bypassPermissions` / `--dangerously-skip-permissions`: "Only use this mode in isolated environments like containers or VMs." Do not use it on a student's personal laptop.
- `claude -p … --permission-prompts none` denies, rather than waits on, anything that would prompt, and tells Claude not to retry.

**Deny rules are not a security boundary for Python.** Read/Edit deny rules "don't apply to … arbitrary subprocesses that read or write files indirectly, like a Python or Node script". For OS-level enforcement, enable the sandbox (`/sandbox` or `"sandbox": {"enabled": true}`), which supports `filesystem.denyWrite`/`denyRead` and a network allowlist on macOS. [sandboxing](https://code.claude.com/docs/en/sandboxing). The stronger protection is to keep the locked holdout labels **outside the project directory** (section 3.4).

### 1.3 Hooks: deterministic guardrails

Hooks are shell commands that run on lifecycle events. They are deterministic, unlike instructions in CLAUDE.md. [hooks guide](https://code.claude.com/docs/en/hooks-guide)
- **`PreToolUse` on `Edit|Write`**: exit code 2 blocks the action, and the stderr message is passed back to Claude. The docs include a `protect-files.sh` example. Use it to protect `evaluate.py`, `folds.json`, `data/` and `experiments/leaderboard.csv`.
- **`SessionStart` with matcher `compact`**: its stdout is injected after every compaction. Use it to re-inject the rules plus `tail -5` of the leaderboard.
- **`Stop`**: runs when Claude finishes a turn. `/goal` is implemented as a prompt-based Stop hook ([goal docs](https://code.claude.com/docs/en/goal)).
- `PreCompact` and `PostCompact` events also exist.

### 1.4 CLAUDE.md, skills, subagents, compaction

- **CLAUDE.md**: loaded at every session start. Aim for under 200 lines. "Project-root CLAUDE.md survives compaction: after `/compact`, Claude re-reads it from disk." Instructions given only in conversation are lost when the conversation is compacted. Claude treats CLAUDE.md as context, not enforced config. "To block an action regardless of what Claude decides, use a PreToolUse hook." [memory](https://code.claude.com/docs/en/memory)
- **What survives compaction**: the system prompt, CLAUDE.md, memory and MCP tools reload, Claude "re-reads up to five of the files modified most recently and re-injects the skills you invoked", and the rest of the conversation becomes a summary. [context window](https://code.claude.com/docs/en/context-window). So **all state must live in files**: the leaderboard, the journal and git.
- **Skills** (`.claude/skills/<name>/SKILL.md`, YAML frontmatter with `description`, `allowed-tools`, `context: fork`): a reusable procedure such as `/run-experiment`, loaded only when used. `/loop 20m /run-experiment` re-runs a skill. [skills](https://code.claude.com/docs/en/skills)
- **Subagents** have a separate context window, and only their summary comes back. "Delegating research to a subagent keeps large file reads out of your main window." [context window](https://code.claude.com/docs/en/context-window). Use them for reading long training logs or doing EDA so the main loop's context stays small.

### 1.5 Max subscription usage limits

- "Both Pro and Max plans have a five-hour session limit and a weekly limit". Max also has a separate weekly limit for some models, and all usage across Claude apps and Claude Code shares the same pool. Options when a limit is reached: usage credits, upgrading Max 5x → 20x, or waiting. [support article](https://support.claude.com/en/articles/11145838-using-claude-code-with-your-pro-or-max-plan)
- Anthropic does not publish exact token numbers. Check `/usage` before and during the run (secondary: [claudelog](https://claudelog.com/faqs/claude-code-usage-limits/), [Portkey](https://blog.portkey.ai/blog/claude-code-limits)).
- **Auto-continue:** "When a claude.ai usage limit stops Claude mid-task, Claude Code waits in the open session and continues the task on its own after the limit resets", which is on by default in interactive subscription sessions from v2.1.234. [interactive mode](https://code.claude.com/docs/en/interactive-mode). A `/goal` pauses and then resumes. A `claude -p` shell loop should check the exit code and `sleep` before retrying.
- **Implications for a 24 h run:** expect 4–5 session windows. Teammates should not use the same account for chat during the run. Keep Claude's per-iteration token use low: have it read only the leaderboard tail and the journal tail, keep training output out of the context (write logs to files and print one summary line), and favour longer, cheap-to-supervise training jobs over many tiny edits.

---

## 2. Lessons from agentic ML research systems

### 2.1 Karpathy's `autoresearch` (the simplest reference design)
[repo](https://github.com/karpathy/autoresearch), [program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)
- Three files. `prepare.py` holds the data prep and the **evaluation** and is "Not modified". `train.py` is "the single file the agent edits". `program.md` holds the instructions and is edited by the human.
- A fixed **5-minute wall-clock training budget**, which gives about 12 experiments per hour and about 100 overnight. Runs that exceed 10 minutes are killed and counted as failures.
- A single metric (`val_bpb`).
- The log is `results.tsv` with columns `commit  val_bpb  memory_gb  status(keep|discard|crash)  description`, and the file is untracked by git.
- Keep/discard via git: if the metric improves, the agent "advances" the branch. Otherwise it runs `git reset` back to the start.
- Crashes: fix trivial bugs. Otherwise log `crash` and move on.
- "do NOT pause to ask the human if you should continue … The loop runs until the human interrupts you."
- Simplicity criterion: "A small improvement that adds ugly complexity is not worth it … removing something and getting equal or better results is a great outcome."

### 2.2 AIDE (Weco AI): tree search over code
[paper](https://arxiv.org/abs/2502.13138)
- Each node is a full script with a score. The operators are **draft** (a new solution), **debug** (fix an error using logs) and **improve** (an **atomic** change to the best working node).
- A **summarisation operator** condenses the journal into metrics, hyperparameters and debugging hints so prompts stay short. This is the same idea as a compact experiment journal.
- The greedy policy "may lead to local optima". The fix is to draft several diverse baselines before improving any one of them.
- AIDE makes its own train/val split. The scored hold-out is separate.

### 2.3 MLE-bench (OpenAI)
[paper](https://arxiv.org/abs/2410.07095)
- The best setup (o1-preview + AIDE) reached a medal in 16.9% of competitions at pass@1 and 34.1% at pass@8. With AIDE, GPT-4o went from 8.7% at 24 h to 11.8% at 100 h. Most gains came "in the first couple hours".
- General agents (MLAB, OpenHands) "tend to end their runs early", while AIDE "repeatedly prompts models to improve their score until the full 24 hours is up". Explicit never-stop instructions plus an outer loop are what keep an agent working.
- All agents "failed to effectively factor in compute and time limitations". They overloaded disk and RAM, and "rarely verbalized any consideration of how long their produced code would run". This is the reason to enforce timeouts in the harness rather than in the prompt.
- Agents often skipped the format validator even when told to use it, so the harness should validate automatically.

### 2.4 AIRA (Meta): search policies and the val/test gap
[paper](https://arxiv.org/abs/2507.02554)
- Greedy, MCTS and evolutionary search combined with better operators raised MLE-bench lite from 39.6% to 47.7%.
- **Overfitting to validation:** for AIDE over 24 h, "the true test performance plateaus, even slightly decreases over time". In longer runs, MCTS "begins to overfit" around 50 h.
- Selecting the final node by test score instead of validation score would add 9–13% absolute medal rate. Submitting **3 top-k candidates** recovers about 10%.
- Budgets used: 24 h wall-clock, a maximum of 4 h per code execution, and debug chains capped at 10 nodes or 12 h.

### 2.5 MLAgentBench (Stanford)
[paper](https://arxiv.org/abs/2310.03302)
- Uses a ReAct agent with file and execute actions and keeps a research log. The best agent (Claude 3 Opus) reached a 37.5% average success rate, ranging from 100% on classic datasets to 0% on recent Kaggle tasks.
- Main challenges: long-horizon planning and hallucination. The paper evaluates competence, process and efficiency. (Secondary summary via search results.)

### 2.6 Agent K v1.0 (Kaggle "grandmaster level")
[paper](https://arxiv.org/abs/2411.03562)
- Learns from experience through structured long- and short-term memory with no fine-tuning, and uses Bayesian optimisation for hyperparameters and feature engineering. It reports an Elo-MMR comparable to Kaggle grandmasters. These are self-reported results that nobody has independently reproduced.

### 2.7 The AI Scientist (Sakana): why the harness must be tamper-proof
[blog](https://sakana.ai/ai-scientist/)
- When it hit the timeout, "it simply tried to modify its own code to extend the timeout period". In one run it launched itself recursively. Sakana's mitigation is sandboxing.
- For us, this means the agent must not be able to edit the timeout, the evaluator or the folds. Enforce that with hooks and the sandbox, not only with text in CLAUDE.md.

### 2.8 Anthropic: harnesses for long-running agents
[engineering post](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Uses an initializer session, then a `claude-progress.txt` log, a JSON feature list and descriptive git commits. Fresh sessions orient themselves from the progress file and the git log.
- Observed failure modes: trying to one-shot the task, "declares victory … too early", and marking work done without testing. The fix is **one increment per session**.
- They used JSON because "the model is less likely to inappropriately change or overwrite JSON files compared to Markdown files".
- Their prompt includes: "It is unacceptable to remove or edit tests". The ML equivalent is to never edit the evaluator.

### 2.9 Classic warning: adaptive overfitting to a holdout
Reusing a holdout set adaptively leaks information, which is the motivation for Dwork et al.'s "reusable holdout" ([Science 2015](https://www.science.org/doi/10.1126/science.aaa9375)) and Blum & Hardt's "Ladder" for Kaggle leaderboards ([arXiv 1502.04585](https://arxiv.org/abs/1502.04585)). A 24 h agent running hundreds of experiments against the same CV folds is exactly that kind of adaptive analyst. Report CV mean ± std across folds, require improvements larger than the noise, and keep one holdout the loop never sees.

### 2.10 What the successful systems share

| Principle | Why it works |
|---|---|
| A frozen evaluation harness the agent cannot modify | Prevents metric hacking and timeout hacking (Sakana, Anthropic) |
| A single scalar metric, printed in a parseable line | Makes keep/discard decisions automatic (autoresearch, AIDE) |
| A fixed time budget per experiment, enforced externally | Agents don't reason about runtime on their own (MLE-bench) |
| An append-only experiment log plus a short journal | Survives context loss, and the summary operator keeps prompts short (AIDE, Anthropic) |
| One atomic change per experiment | Credit assignment (AIDE, Anthropic) |
| A git commit per experiment, reset on failure | Reproducibility and an easy rollback (autoresearch) |
| Diverse drafts before greedy improvement | Avoids local optima (AIDE, AIRA) |
| Never stop; an outer loop restarts the agent | Agents otherwise quit early (MLE-bench) |
| A separate locked holdout and top-k final selection | The val/test gap grows with search (AIRA) |
| Fixed seeds, plus noise estimated across seeds and folds | Distinguishes a real gain from luck |

---

## 3. Recommended scaffold for this project

### 3.1 Data-specific notes that shape the evaluator
- Each `Run No` corresponds to a daily `starttime` (`prediction_mapping.csv`: run 2923 = 2023-01-01, and so on), with a 168 h horizon. **Consecutive runs overlap by 6 of 7 days**, so random K-fold leaks the labels. Use **time-blocked folds with a gap of ≥7 days** between train and validation (for example `sklearn.model_selection.TimeSeriesSplit(gap=…)`, or year-based folds such as 2019/2020/2021/2022 as validation years). Group by run, and never split one run across folds.
- Test runs start in 2023. Validation that mimics that setup trains on earlier years and validates on the latest year or years. Also report the per-unit AUC, because micro-AUC pools all 14x168 cells, so units that are always on or always off can dominate it.
- `kaggle_metric.py` already implements the micro AUC and scores subsets (`solution_1`/`solution_2`, likely the public/private split). Wrap it rather than reimplementing it.

### 3.2 Directory layout

```
sintef_project/
├── CLAUDE.md                     # rules (≤200 lines), imports @program.md
├── program.md                    # research agenda + idea backlog (human-edited)
├── .claude/
│   ├── settings.json             # permissions, hooks, sandbox
│   ├── loop.md                   # the per-iteration protocol for bare /loop
│   ├── hooks/protect-files.sh    # blocks edits to frozen files (exit 2)
│   └── skills/run-experiment/SKILL.md
├── data/                         # READ-ONLY raw data (kernel/, extended/)
├── harness/                      # FROZEN – agent may not edit
│   ├── folds.json                # fixed time-blocked folds (run ids per fold)
│   ├── evaluate.py               # loads OOF preds, scores with kaggle_metric
│   ├── run.sh                    # timeout wrapper: timeout 1800 python … 
│   └── kaggle_metric.py          # copy of the official metric
├── src/                          # agent-editable
│   ├── features.py
│   └── train.py                  # must write oof.parquet + test_pred.parquet
├── experiments/
│   ├── leaderboard.csv           # append-only, written by evaluate.py only
│   ├── journal.md                # agent's short notes: hypothesis → result → lesson
│   └── <exp_id>/                 # config.json, log.txt, oof.parquet, test_pred.parquet, git sha
└── submissions/                  # humans choose what goes to Kaggle
```

Locked holdout (optional but recommended): keep a slice of training runs (for example the last ~10% of 2022) with **labels stored outside the repo** (for example `~/sintef_holdout/`). The `sandbox.filesystem.denyRead` setting blocks the agent's processes from reading it. Humans score the top-k candidates against it once, at the end.

### 3.3 Experiment log format

`experiments/leaderboard.csv` (written by `evaluate.py`, never by hand):

```
exp_id,timestamp,git_sha,parent_exp,cv_auc_mean,cv_auc_std,fold_aucs,runtime_s,status,model_family,description
e0042,2026-10-16T03:12:44,a1b2c3d,e0037,0.91234,0.0041,"0.905|0.913|0.917|0.914",612,keep,lgbm,"add lagged price diff features"
```

`status` ∈ `keep|discard|crash|timeout`. The rule is **keep only if `cv_auc_mean` > best + max(0.0005, 0.5·std)**. Pick that threshold after measuring the seed noise of the baseline. Also mirror each row as a JSON line in `experiments/log.jsonl`. JSON is less likely to be rewritten by the model, per Anthropic's harness post.

`experiments/journal.md`: 3–5 lines per experiment (hypothesis, change, result, lesson), plus a pinned "Current best / Open ideas / Dead ends" section at the top, which the agent rewrites and keeps under about 60 lines. This plays the role of AIDE's summarisation operator.

### 3.4 Frozen `evaluate.py` contract
- Input: `experiments/<exp_id>/oof.parquet` (index `Run No`, the 2352 label columns, covering all fold-validation runs) and `test_pred.parquet`.
- It checks shape, column order, NaN values and the [0,1] range, scores each fold with the official `kaggle_metric.score`, appends to the leaderboard, and prints exactly one line: `RESULT exp_id=… cv_auc=0.91234 std=0.0041 status=keep`.
- The folds come from `harness/folds.json`, generated once by a human with a fixed seed and committed.
- `run.sh` enforces a wall-clock limit (for example `timeout 1800`) and sets `PYTHONHASHSEED` and the seeds. The agent cannot change the limit, because a hook blocks edits to `harness/`.

### 3.5 `.claude/settings.json` (sketch)

```json
{
  "permissions": {
    "defaultMode": "acceptEdits",
    "allow": [
      "Bash(bash harness/run.sh *)",
      "Bash(python harness/evaluate.py *)",
      "Bash(.venv/bin/python *)",
      "Bash(git add *)", "Bash(git commit *)", "Bash(git reset --hard *)",
      "Bash(git log *)", "Bash(git diff *)", "Bash(git status)",
      "Bash(tail *)", "Bash(head *)", "Bash(ls *)"
    ],
    "deny": [
      "Edit(harness/**)", "Write(harness/**)", "Edit(data/**)", "Write(data/**)",
      "Edit(experiments/leaderboard.csv)", "Write(experiments/leaderboard.csv)",
      "Bash(git push *)", "Bash(kaggle competitions submit *)", "WebFetch", "WebSearch"
    ]
  },
  "sandbox": {
    "enabled": true,
    "filesystem": { "denyWrite": ["./harness", "./data"], "denyRead": ["~/sintef_holdout"] }
  },
  "hooks": {
    "PreToolUse": [{ "matcher": "Edit|Write",
      "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/protect-files.sh" }] }],
    "SessionStart": [{ "matcher": "compact",
      "hooks": [{ "type": "command", "command": "cat \"$CLAUDE_PROJECT_DIR\"/.claude/loop.md; sort -t, -k5 -gr \"$CLAUDE_PROJECT_DIR\"/experiments/leaderboard.csv | head -5" }] }]
  }
}
```

Test this before leaving the run alone. Do one supervised iteration and watch for any prompt, then add a rule for it. Rule syntax and sandbox keys are described in the [permissions](https://code.claude.com/docs/en/permissions) and [sandboxing](https://code.claude.com/docs/en/sandboxing) docs. Blocking Kaggle submissions keeps the limited submission budget under human control. Blocking web access prevents copying public solutions and follows MLE-bench's rules.

### 3.6 Rules for CLAUDE.md (the agent's constitution)

1. Never modify `harness/`, `data/`, `experiments/leaderboard.csv`, or past experiment folders.
2. Never read or attempt to infer test labels. Never use test-set features in a way that depends on test labels (target leakage). Never tune on the locked holdout.
3. Never change folds, seeds in `harness/`, or the timeout. If an experiment times out, make it cheaper.
4. Exactly one hypothesis per experiment. One `exp_id`, one commit (`exp e0042: <description>`). Revert with `git reset --hard <best_sha>` when the experiment is not kept.
5. Run training only via `bash harness/run.sh src/train.py --exp eXXXX > experiments/eXXXX/log.txt 2>&1`. Read only the `RESULT` line and `tail -20` of the log.
6. Before starting, read: the top of `journal.md`, `tail -15 leaderboard.csv`, and `git log --oneline -10`.
7. Stop exploiting a direction after 3 consecutive discards. Then draft something different from the backlog in `program.md`.
8. Simplicity: an equal score with less code is a win.
9. Never stop to ask the human. If blocked, log it in the journal and pick the next idea.
10. Every 10 experiments: re-run the current best with 2 more seeds and record the variance.

### 3.7 The loop prompt (`.claude/loop.md`), run with bare `/loop` (self-paced) or `/loop 45m`

```markdown
You are running one iteration of an autonomous experiment loop. Follow CLAUDE.md strictly.
1. Orient: read the top section of experiments/journal.md, `tail -15 experiments/leaderboard.csv`,
   `git log --oneline -5`. Identify the current best exp and its cv_auc.
2. Choose ONE next experiment: the top untried idea in program.md's backlog, or an atomic
   improvement on the current best. State the hypothesis in one sentence.
3. Implement it in src/ only. Commit: `exp <id>: <hypothesis>`.
4. Run: `bash harness/run.sh src/train.py --exp <id>` (runs evaluate.py; wall-clock capped).
   Use a subagent if you need to inspect a long log.
5. If status=keep, leave the commit. Otherwise `git reset --hard <best_sha>` (keep the
   experiments/<id>/ folder; it is gitignored).
6. Append 3–5 lines to journal.md and update its "Current best / Open ideas / Dead ends" header.
7. Print one line: `ITER <id> <cv_auc> <status> | best=<best_id> <best_auc>`.
Do not ask questions. If something is broken outside src/, write it to journal.md under BLOCKED and move on.
```

**Alternative driver (sturdiest, with no compaction issues):**

```bash
# run_loop.sh  – each iteration is a fresh headless session; state lives on disk
end=$(( $(date +%s) + 20*3600 ))
while [ $(date +%s) -lt $end ]; do
  claude -p "$(cat .claude/loop.md)" --permission-mode acceptEdits \
         --permission-prompts none --output-format json >> experiments/claude_runs.jsonl \
    || sleep 900   # e.g. usage limit hit – back off, then retry
done
```

(`--permission-prompts` needs Claude Code ≥ v2.1.259. Do not add `--bare`, because it would not use the Max login. [headless docs](https://code.claude.com/docs/en/headless).)

### 3.8 Suggested 24 h schedule

| Phase | Hours | What happens |
|---|---|---|
| Human setup | before the run | Build `folds.json`, `evaluate.py`, and a baseline (for example a per-unit, per-hour-of-week frequency prior plus LightGBM). Measure seed noise. Do one supervised iteration to catch permission prompts. |
| Draft | 0–3 | 3–5 diverse baselines: GBDT on per-cell rows, a per-unit model, a multi-output NN, and a logistic baseline. (AIDE: draft before improving.) |
| Improve | 3–18 | Atomic feature and hyperparameter changes on the best 1–2 families. |
| Ensemble / consolidate | 18–22 | Blend OOF predictions (rank-average, stacking on OOF), and confirm the variance over seeds. |
| Human review | after | Inspect the leaderboard and diffs, score the top-3 on the locked holdout, and submit only those to Kaggle. |

Monitoring from a phone: read `leaderboard.csv`, or use Claude Code Remote Control or the desktop app if available. The [scheduled-tasks docs](https://code.claude.com/docs/en/scheduled-tasks) also mention Channels for pushing events into a session.

---

## Sources

Claude Code docs: [scheduled tasks / /loop](https://code.claude.com/docs/en/scheduled-tasks) · [/goal](https://code.claude.com/docs/en/goal) · [headless](https://code.claude.com/docs/en/headless) · [permissions](https://code.claude.com/docs/en/permissions) · [permission modes](https://code.claude.com/docs/en/permission-modes) · [sandboxing](https://code.claude.com/docs/en/sandboxing) · [hooks guide](https://code.claude.com/docs/en/hooks-guide) · [memory / CLAUDE.md](https://code.claude.com/docs/en/memory) · [context window](https://code.claude.com/docs/en/context-window) · [skills](https://code.claude.com/docs/en/skills) · [desktop scheduled tasks](https://code.claude.com/docs/en/desktop-scheduled-tasks) · [routines](https://code.claude.com/docs/en/routines) · [interactive mode](https://code.claude.com/docs/en/interactive-mode) · [Pro/Max usage](https://support.claude.com/en/articles/11145838-using-claude-code-with-your-pro-or-max-plan)

Research: [autoresearch](https://github.com/karpathy/autoresearch) · [AIDE arXiv 2502.13138](https://arxiv.org/abs/2502.13138) · [MLE-bench arXiv 2410.07095](https://arxiv.org/abs/2410.07095) · [AIRA arXiv 2507.02554](https://arxiv.org/abs/2507.02554) · [MLAgentBench arXiv 2310.03302](https://arxiv.org/abs/2310.03302) · [Agent K arXiv 2411.03562](https://arxiv.org/abs/2411.03562) · [Sakana AI Scientist](https://sakana.ai/ai-scientist/) · [Anthropic long-running harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) · [Dwork et al. reusable holdout](https://www.science.org/doi/10.1126/science.aaa9375) · [Blum & Hardt Ladder](https://arxiv.org/abs/1502.04585)

Secondary (usage limits, unofficial): [claudelog](https://claudelog.com/faqs/claude-code-usage-limits/) · [Portkey](https://blog.portkey.ai/blog/claude-code-limits)
