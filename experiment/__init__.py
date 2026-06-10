"""Experimental engine + bench harness.

`experiment.engine` is the active iteration target — change here, not rag/engine.py.
`experiment.bench` runs question.xlsx and reports hit@5 vs the baseline target (70%).

After a variant reaches 70%, the change graduates to rag/engine.py + contract test.
"""
