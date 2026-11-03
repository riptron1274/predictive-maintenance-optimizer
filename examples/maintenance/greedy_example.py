
from pathlib import Path
import sys

example_dir = (
    Path(__file__).resolve().parent
    / "examples"
    / "maintenance"
)

sys.path.insert(0, str(example_dir))

from equipment import Equipment
from optimization import select_maintenance


machines = [
    Equipment("A", 1.0, 12, 6, 1.0),
    Equipment("B", 1.0, 9, 5, 1.0),
    Equipment("C", 1.0, 9, 5, 1.0),
]

selected = select_maintenance(machines, budget=10)

print([machine.name for machine in selected])

total_benefit = sum(
    machine.expected_avoided_loss()
    for machine in selected
)

print(total_benefit)
