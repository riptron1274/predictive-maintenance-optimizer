from equipment import Equipment


def print_report(
    selected: list[Equipment],
    budget: float
) -> None:

    total_cost = sum(
        machine.maintenance_cost
        for machine in selected
    )

    total_benefit = sum(
        machine.expected_avoided_loss()
        for machine in selected
    )

    print("MAINTENANCE PLAN")
    print("----------------")
    print(f"Number of machines selected: {len(selected)}")
    for machine in selected:
        print(f"Machine selected for maintenance is {machine.name}")

    print(f"Total cost is €{total_cost:,.2f}")
    print(f"Expected avoided loss is €{total_benefit:,.2f}")
    print(f"Unused budget is €{budget - total_cost:,.2f}")
