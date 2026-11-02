from dataclasses import dataclass

@dataclass
class Equipment:
    name: str
    failure_probability: float
    failure_cost: float
    maintenance_cost: float
    maintenance_effectiveness: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Equipment name cannot be empty!")

        if not 0 <= self.failure_probability <= 1:
            raise ValueError("Failure probability must be between 0 and 1!")

        if self.failure_cost < 0:
            raise ValueError("Failure cost must be positive!")

        if self.maintenance_cost <= 0:
            raise ValueError("Maintenance cost must be positive!")

        if not 0 <= self.maintenance_effectiveness <= 1:
            raise ValueError("Maintenance effectiveness must be between 0 and 1!")

    def expected_avoided_loss(self) -> float:
        return (
            self.failure_probability
            * self.failure_cost
            * self.maintenance_effectiveness
        )

    def benefit_cost_ratio(self) -> float:
        return (
            self.expected_avoided_loss()
            / self.maintenance_cost
        )
