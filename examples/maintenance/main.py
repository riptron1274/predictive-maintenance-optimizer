from pathlib import Path

from data_process import load_equipment
from optimization import select_maintenance
from report import print_report


def main() -> None:

    file_path = Path(__file__).resolve().parent / "equipment.csv"
    equipment = load_equipment(file_path)
    budget = 5000 #Budget available
    selected = select_maintenance(equipment, budget)
    print_report(selected, budget)

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Application error: {error}")
