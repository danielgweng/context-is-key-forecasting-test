from google.cloud import bigquery
import pandas as pd

from ..base import UnivariateCRPSTask

class GMVPredictionTask(UnivariateCRPSTask):
    """
    A task where the model is given a time series of GMV and has to predict the next 365 days of GMV.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.3"  # Modification will trigger re-caching

    def random_instance(self):
        # Initialize BigQuery client
        client = bigquery.Client()
        
        # Query to get time series data for specific segment
        query = """
        SELECT 
            ds,
            SUM(gmv_local) as gmv_local
        FROM `sdp-prd-finance-data-science.intermediate.gmv_forecast_actuals_shop_currency`
        WHERE 
            fpa_region_group = 'United States'
            AND merchant_type = 'Plus'
            AND sales_channel = 'Online+'
            AND currency = 'USD'
        GROUP BY ds
        ORDER BY ds
        """
        
        # Run the query and convert to pandas
        df = client.query(query).to_dataframe()
        df['ds'] = pd.to_datetime(df['ds'])
        df.set_index('ds', inplace=True)
        
        # Split into history and future - last 365 days as future
        future_series = df.iloc[-365:]
        history_series = df.iloc[-365*4:-365]

        # Convert to timestamp for consistency
        # future_series.index = future_series.index.to_timestamp()
        # history_series.index = history_series.index.to_timestamp()

        background = """
        This is Shopify's daily Gross Merchandise Volume (GMV) value for the United States region, Plus merchant type, Online+ sales channel, in USD currency.

        GMV tends to trend upwards over time, but has multiple seasonal components:
        - weekly seasonality, Saturday and Sunday tend to be the slowest days of the week
        - yearly seasonality, November and December are the busiest months of the year due to Black Friday Cyber Monday (BFCM) and the Christmas holiday season
        - holiday seasonality, Black Friday Cyber Monday (BFCM) starts on the fourth Friday of November and ends on the Monday after Thanksgiving. However, we typically see a ramp up in GMV during the week leading up to BFCM.
        - BFCM in 2024 starts 5 days later than it did in 2023.
        """

        # Instantiate the class variables
        self.past_time = history_series
        self.future_time = future_series
        self.constraints = None
        self.background = background

__TASKS__ = [GMVPredictionTask]
__CLUSTERS__ = []
