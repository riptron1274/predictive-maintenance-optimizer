import csv
from pathlib import Path
from equipment import Equipment


def load_equipment(file_path: Path) -> list[Equipment]:

    machines = []

    with file_path.open("r", newline="", encoding="utf-8") as file:
        """"Reading CSV file"""
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=2):
            try:
                machine = Equipment(
                    name=row["name"],
                    failure_probability=float(row["failure_probability"]),
                    failure_cost=float(row["failure_cost"]),
                    maintenance_cost=float(row["maintenance_cost"]),
                    maintenance_effectiveness=float(
                        row["maintenance_effectiveness"]
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid equipment data on row {row_number}: {error}"
                ) from error

            machines.append(machine)

    return machines
