import importlib.resources as pkg_resources
import pandas as pd

from google.cloud import bigquery
from ml_package import sql_templates
from .gcp_conn import GCPConnection



class BigQueryOperations(GCPConnection):
    def __init__(self, project_id):
        super().__init__()
        self.bg_client = bigquery.Client(
            project=project_id,
        )

    def get_warehouses(self, country_code: str) -> pd.DataFrame:
        warehouses = self.read_bigquery(
            pkg_resources.read_text(sql_templates, "warehouse.sql"),
            verbose=True,
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "country_code", "STRING", country_code),
            ],
        )
        return warehouses

    def get_customer_order_data(
        self,
        global_entity_id: str,
        # city_id: str,
        # country_code: str,
        date_from: str,
        date_to: str,
        round_off: int = 4,
    ) -> pd.DataFrame:
        products = self.read_bigquery(
            pkg_resources.read_text(sql_templates, "customer_orders.sql"),
            verbose=True,
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "global_entity_id", "STRING", global_entity_id
                ),
                # bigquery.ScalarQueryParameter("city_id", "STRING", city_id),
                # bigquery.ScalarQueryParameter(
                #     "country_code", "STRING", country_code),
                bigquery.ScalarQueryParameter("date_from", "DATE", date_from),
                bigquery.ScalarQueryParameter("date_to", "DATE", date_to),
                bigquery.ScalarQueryParameter("round_off", "INT64", round_off),
            ],
        )
        return products

    def execute_query(self, sql, query_parameters=None) -> pd.DataFrame:
        if query_parameters is None:
            query_parameters = []
        job_config = self.bg_client.QueryJobConfig(
            query_parameters=query_parameters)

        query_job = self.bg_client.query(sql, job_config=job_config)
        results = query_job.result()
        for row in results:
            print(f"{row.order_id} : {row.global_entity_id} ")
        df = results.to_dataframe()
        return df

    def read_bigquery(
        self,
        query: str,
        parse_dates: list = None,
        verbose=False,
        query_parameters=None,
    ) -> pd.DataFrame:
        """
        Load a query from BigQuery.
        query: query-string or file path
        """

        if query.endswith(".data_store"):
            query_string = open(query).read().replace("%", "%%")
        else:
            query_string = query

        if query_parameters is None:
            query_parameters = []

        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)

        job = self.bg_client.query(query_string, job_config=job_config)
        if verbose:
            # logger.info("bigQuery job done, downloading...")
            print("bigQuery job done, downloading...")
        result = job.result()
        df = result.to_dataframe()

        return df