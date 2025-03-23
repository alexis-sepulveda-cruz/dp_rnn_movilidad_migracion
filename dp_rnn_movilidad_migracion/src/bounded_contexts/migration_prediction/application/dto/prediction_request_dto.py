import pandas as pd


class PredictionRequestDTO:
    """DTO para solicitar una predicción."""

    def __init__(self, state: str, temporal_data: pd.DataFrame,
                 static_data: pd.DataFrame, future_years: int = 5,
                 mc_samples: int = 100, visualize: bool = False):
        self.state = state
        self.temporal_data = temporal_data
        self.static_data = static_data
        self.future_years = future_years
        self.mc_samples = mc_samples
        self.visualize = visualize