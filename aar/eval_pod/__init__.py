"""Eval-pod side: loads a submitted model and scores it against the secret
benchmark suite. Runs as a standalone process (locally for the toy; on an
ephemeral GPU pod in production). Nothing here is importable by the research
side — it lives behind the isolation boundary.
"""
