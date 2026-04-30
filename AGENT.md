# Agent Coding Style Guide

Write code like a careful human engineer working inside this repository. Keep the code compact, readable, and intentional.

## Core Style

- Match the style already used in the file you are editing.
- Prefer compact spacing for simple expressions: use `a=b`, `x+=1`, `foo(bar)` when the local file follows that style.
- Do not add artificial blank lines to make code look bigger.
- Do not align assignments or columns with extra spaces.
- Do not rewrite working code just to make it look more "AI formatted".
- Keep changes narrowly scoped to the requested task.

## Comments

- Add comments only when they explain a decision, tradeoff, mathematical idea, non-obvious edge case, or experiment assumption.
- Do not comment obvious statements.
- Do not add section-banner comments unless the file already uses them.
- Prefer one precise sentence over long explanatory blocks.

Bad:

```python
# Assign the value to x
x=1

# Loop through each item
for item in items:
    process(item)
```

Good:

```python
# Scale communication before solving; raw byte counts make some solvers unstable.
cost_scale=max(float(np.max(np.abs(costs_np))),1.0)
```

## Formatting

- Avoid unnatural gaps between related lines.
- Keep imports, constants, helpers, and main logic organized, but do not over-separate tiny blocks.
- Do not introduce decorative whitespace.
- Let code density follow complexity: simple code can be tight; complex logic can breathe.
- If a formatter exists for the language, follow the formatter instead of inventing a style.

Preferred compact style in this repo:

```python
weights=sizes.astype("float64")/sizes.sum()
if cfg.cvar_alpha<=0:
    return weights
```

Avoid:

```python
weights = sizes.astype("float64") / sizes.sum()

if cfg.cvar_alpha <= 0:
    return weights
```

## Implementation Judgment

- Prefer existing functions, data structures, and experiment patterns.
- Do not add abstractions unless they remove real duplication or clarify a real concept.
- Do not add defensive fallbacks for impossible states unless the code handles external input.
- Keep research code reproducible: save configs, seeds, outputs, and assumptions when experiments run.
- When changing experiment logic, preserve the ability to trace results back to raw CSVs, model artifacts, and report tables.

## Reports and Explanations

- Reports should explain why each step exists, not only what was executed.
- Do not pad reports with font changes, spacing tricks, or filler text.
- If a report must be long, make it long through real content: methods, assumptions, math, interpretation, limitations, and reproducibility details.

## Final Checks

Before finishing code changes:

- Run the smallest useful validation command.
- Check lint/type errors when available.
- Verify generated artifacts exist if the task creates reports, plots, or models.
- Do not commit or push unless explicitly asked.
