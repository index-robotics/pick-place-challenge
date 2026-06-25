"""Console entry points for the pick-and-place challenge.

Our tasks register with mjlab via the ``mjlab.tasks`` entry-point group (see
``pyproject.toml``): on ``import mjlab`` they are auto-imported, so mjlab's own
``play``/``train`` CLIs can see ``Mjlab-PlaceBall-Franka-*`` without any
wrappers. Run them directly, e.g.::

    uv run play Mjlab-PlaceBall-Franka-State-v0 --agent random
"""

from __future__ import annotations


def list_envs() -> None:
    """Print the registered Franka pick-and-place task IDs."""
    import pick_place_challenge.task  # noqa: F401  (registration side effect)
    import mjlab.tasks  # noqa: F401  (built-in tasks too)
    from mjlab.tasks.registry import list_tasks

    for task_id in list_tasks():
        marker = "*" if "Franka" in task_id else " "
        print(f"{marker} {task_id}")
