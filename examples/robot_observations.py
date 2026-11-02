"""A robot's episodic memory: every observation stored verbatim with time and place, recalled by question.

Positions are kept in the text, so they are searchable as words ("kitchen", "dock 3") and shown to the planner
verbatim. Run with fake models to see the flow without downloading weights:  python examples/robot_observations.py
"""
import muster
from muster.models import Models

store = muster.Store.open("robot.muster", models=Models.fake(32))   # drop models= to use the real ones
log = [
    ("2026-10-20T08:01", "kitchen", "Red mug on the kitchen table, next to the kettle."),
    ("2026-10-20T08:05", "hallway", "Charging dock 3 is blocked by a cardboard box."),
    ("2026-10-20T09:40", "kitchen", "Red mug moved to the dish rack."),
    ("2026-10-20T10:15", "office",  "Ana asked for her laptop charger; it is in the second drawer."),
]
for t, place, seen in log:
    store.add({"speaker": "camera", "time": t, "text": "[%s] %s" % (place, seen)})
print(store.context("Where is the red mug now?").text)
