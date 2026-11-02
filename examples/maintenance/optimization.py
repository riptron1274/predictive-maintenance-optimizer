from equipment import Equipment

def select_maintenance(
    equipment: list[Equipment],
    budget: float
    ) -> list[Equipment]:

    if budget < 0:
        raise ValueError("Budget must be positive!")

    ranked = sorted(
        equipment,
        key=lambda machine: machine.benefit_cost_ratio(),
        reverse=True
    )

    selected: list[Equipment] = []
    remaining_budget = budget

    for machine in ranked:
        if machine.maintenance_cost <= remaining_budget:
            selected.append(machine)
            remaining_budget -= machine.maintenance_cost

    return selected
