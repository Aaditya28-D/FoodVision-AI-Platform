class EarlyStopping:
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        improved = False

        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            return False

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop