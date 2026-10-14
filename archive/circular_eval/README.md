# ⚠️ INVALID: circular evaluation (kept only as a record)

`relabel_dataset.py` re-labelled the dataset by putting each image into the folder that
**the pipeline itself predicted** (`ai/data/relabeled/`). `scripts/bias_audit.py` was then
run against those folders and reported **97.1% "undertone accuracy"**
(`audit_results_relabeled/`, git-ignored locally).

That number is meaningless: it grades the pipeline against its own answers. It isn't 100%
only because the code changed between the relabelling run and the audit run. As committed,
the script couldn't even run (it called an undefined `get_new_label`).

Also, the *original* folder labels in `ai/data/raw/` were not human labels. They were
produced automatically by an older colour heuristic (`backend/undertone_analyzer.py`, via
the deleted `ai/auto_label_all.py`). So no number computed against any of these folders
measures accuracy against reality; at best it measures agreement between two heuristics.

For a real evaluation, see `scripts/eval.py` and `eval/README.md` (hand-labelled images).
