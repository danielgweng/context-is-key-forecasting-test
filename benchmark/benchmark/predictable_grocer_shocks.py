import json
import os
import pandas as pd

from .base import UnivariateCRPSTask
from .data.dominicks import (
    download_dominicks,
    DOMINICK_JSON_PATH,
    DOMINICK_CSV_PATH,
)
from .utils import get_random_window_univar


class PredictableGrocerPersistentShockUnivariateTask(UnivariateCRPSTask):
    """
    A task where the time series contains spikes that are predictable based on the
    contextual information provided with the data. The spikes should be reflected in
    the forecast.
    Note: this does NOT use the Monash dominick's dataset, which is transformed with no
    meaningful context.
    Context: synthetic (GPT-generated then edited)
    Series: modified
    Dataset: Dominick's grocer dataset (daily)
    Parameters:
    -----------
    fixed_config: dict
        A dictionary containing fixed parameters for the task
    seed: int
        Seed for the random number generator
    GROCER_SALES_INFLUENCES_PATH: str
        Path to the JSON file containing the sales influences.
    DOMINICK_GROCER_SALES_PATH: str
        Path to the filtered Dominick's grocer dataset.
        Filtered for a subset of products for which we generated influences.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):
        self.init_data()
        self.dominick_grocer_sales_path = DOMINICK_CSV_PATH
        self.grocer_sales_influences_path = DOMINICK_JSON_PATH
        with open(self.grocer_sales_influences_path, "r") as file:
            self.influences = json.load(file)

        super().__init__(seed=seed, fixed_config=fixed_config)

    def init_data(self):
        """
        Check integrity of data files and download if needed.

        """
        if not os.path.exists(DOMINICK_JSON_PATH):
            raise FileNotFoundError("Missing Dominick json file.")
        if not os.path.exists(DOMINICK_CSV_PATH):
            download_dominicks()

    def random_instance(self):
        dataset = pd.read_csv(self.dominick_grocer_sales_path)
        dataset["date"] = pd.to_datetime(dataset["datetime"])
        dataset = dataset.set_index("date")

        sales_categories = ["grocery", "beer", "meat"]
        stores = dataset["store"].unique()

        self.prediction_length = self.random.randint(7, 30)

        for counter in range(100000):
            # pick a random sales category and store
            sales_category = self.random.choice(sales_categories)
            store = self.random.choice(stores)

            # select a random series
            series = dataset[dataset["store"] == store][sales_category]

            if (series == 0).mean() > 0.5:
                continue

            # select a random window
            history_factor = self.random.randint(3, 7)
            if len(series) > (history_factor + 1) * self.prediction_length:
                window = get_random_window_univar(
                    series,
                    prediction_length=self.prediction_length,
                    history_factor=history_factor,
                    random=self.random,
                )
                break  # Found a valid window, stop the loop
        else:
            raise ValueError("Could not find a valid window.")

        window = self.mitigate_memorization(window)

        # extract the history and future series
        history_series = window.iloc[: -self.prediction_length]
        future_series = window.iloc[-self.prediction_length :]
        ground_truth = future_series.copy()

        # choose an influence and a relative impact from the influence
        shock_delay_in_days = self.random.randint(0, self.prediction_length - 1)
        shock_duration = self.prediction_length - shock_delay_in_days + 1

        direction = self.random.choice(["positive", "negative"])
        size = self.random.choice(["small", "medium", "large"])
        influence_info = self.influences[sales_category][direction][size]
        impact_range = influence_info["impact"]
        self.min_magnitude, self.max_magnitude = map(
            lambda x: int(x.strip("%")), impact_range.split("-")
        )
        impact_magnitude = self.random.randint(self.min_magnitude, self.max_magnitude)

        # apply the influence to the future series
        future_series[shock_delay_in_days : shock_delay_in_days + shock_duration] = (
            self.apply_influence_to_series(
                future_series[
                    shock_delay_in_days : shock_delay_in_days + shock_duration
                ],
                impact_magnitude,
                direction,
            )
        )

        self.min_magnitude = self.min_magnitude
        self.max_magnitude = self.max_magnitude
        self.impact_magnitude = impact_magnitude
        self.direction = direction
        self.ground_truth = ground_truth

        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = None
        self.scenario = self.get_scenario_context(shock_delay_in_days, influence_info)

        self.region_of_interest = slice(
            shock_delay_in_days, shock_delay_in_days + shock_duration
        )

    def mitigate_memorization(self, window):
        """
        Mitigate memorization by adding a random noise to the series
        Parameters:
        -----------
        window: pd.Series
            The series to mitigate memorization
        Returns:
        --------
        window: pd.Series
            The series with mitigated memorization
        """
        window = window.copy()
        window *= 2

        # update the year of the timesteps to map the lowest year of the window to 2024, and increment accordingly
        min_year = window.index.min().year

        def map_year(year):
            return 2024 + (year - min_year)

        window.index = window.index.map(lambda x: x.replace(year=map_year(x.year)))

        return window

    def get_shock_description(self, shock_delay_in_days, influence_info):
        return influence_info["influence"].replace(
            "{time_in_days}", str(shock_delay_in_days)
        )

    def apply_influence_to_series(self, series, relative_impact, direction):
        """
        Apply a relative impact to a series
        Parameters:
        -----------
        series: pd.Series
            The series to apply the impact to.
        relative_impact: int
            The relative impact to apply
        direction: str
            The direction of the impact
        Returns:
        --------
        series: pd.Series
            The series with the applied impact
        """
        if direction == "positive":
            series += series * (relative_impact / 100)
        else:
            series -= series * (relative_impact / 100)

        return series

    def get_scenario_context(self, shock_delay_in_days, influence_info):
        """
        Get the context of the event.
        Returns:
        --------
        context: str
            The context of the event, including the influence and the relative impact.

        """
        relative_impact = self.impact_magnitude
        if self.direction == "negative":
            relative_impact = self.impact_magnitude * -1

        shock_description = influence_info["influence"].replace(
            "{time_in_days}", str(shock_delay_in_days)
        )
        shock_description = shock_description.replace(
            "{impact}", str(relative_impact) + "%"
        )
        return shock_description

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        return 7


__TASKS__ = [PredictableGrocerPersistentShockUnivariateTask]
