# Academic paper working draft

- `PLAN.md`: research framing, section plan, planned figures/tables and required evidence.
- `main.tex`: journal-style manuscript skeleton; uses IEEEtran if installed, otherwise a two-column article fallback. It is not yet a selected publisher template.
- `abstract.tex`: provisional methods-focused abstract, without invented performance or hardware results.
- `introduction.tex`: the opening Introduction; the final contribution statement remains pending.
- `references.bib`: the three supplied anchor papers, with verified bibliographic metadata.

Build from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The final submission requires a selected venue, author/affiliation details, a broader related-work search, completed communication results, qualified RTL results and measured hardware evidence appropriate to its claims. No “first,” NN superiority, full-rate operation or power-saving claim is established by this draft. Existing equalizer code work is separate from this writing-only update.
