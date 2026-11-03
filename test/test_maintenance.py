
from pathlib import Path
import sys

import pytest

# Make the existing example modules available to the tests.
EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples" / "maintenance"
)

sys.path.insert(0, str(EXAMPLE_DIR))

from equipment import Equipment # type: ignore[reportMissingImports]
from optimization import select_maintenance  # type: ignore[reportMissingImports]
from data_process import load_equipment # type: ignore[reportMissingImports]


def test_expected_avoided_loss():
    motor = Equipment(
        "Motor A", 0.4, 12000, 2500, 0.75
    )

    assert motor.expected_avoided_loss() == pytest.approx(3600)


def test_invalid_probability():
    with pytest.raises(ValueError):
        Equipment("Motor A", 1.5, 12000, 2500, 0.75)


def test_invalid_maintenance_cost():
    with pytest.raises(ValueError):
        Equipment("Motor A", 0.4, 12000, 0, 0.75)


def test_equipment_data_loads():
    machines = load_equipment(EXAMPLE_DIR / "equipment.csv")

    assert len(machines) == 4


def test_maintenance_selection():
    machines = load_equipment(EXAMPLE_DIR / "equipment.csv")

    selected = select_maintenance(machines, budget=5000)

    selected_names = [machine.name for machine in selected]
    total_cost = sum(machine.maintenance_cost for machine in selected)

    assert selected_names == ["Pump D", "Motor A", "Pump B"]
    assert total_cost == pytest.approx(4800)


def test_negative_budget():
    with pytest.raises(ValueError):
        select_maintenance([], budget=-100)
