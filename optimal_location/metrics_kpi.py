from optimal_location.utils import *
import pandas as pd
from typing import Union


def select_polygons_based_on_input_params(
    warehouse_polygon_df: pd.DataFrame, input_params: dict
) -> pd.DataFrame:
    """
    Returns polygons based on Input Parameters in DataFrame format .

    Parameters:
           warehouse_polygon_df (DataFrame): 1 - T minutes Polygon for each store location
           input_params(dict) : Trigger input

    Returns:
            polygons_based_on_driving(DataFrame): 1 - T minutes Polygon for each store location based on
                                                  input_params
    """
    drive_mode_selected = input_params["driving_mode"]
    drive_time_selected = input_params["driving_time"]

    # Filter Low DriveTime for absence of Polygons
    polygons_based_on_driving = warehouse_polygon_df[
        (warehouse_polygon_df["driving_mode"] == drive_mode_selected)
        & (warehouse_polygon_df["driving_time"] <= drive_time_selected)
        & (warehouse_polygon_df["driving_time"] > 3)
    ]
    return polygons_based_on_driving


def update_metrics(
    metric_df: pd.DataFrame, store_level_new_metrics: dict
) -> pd.DataFrame:
    """
    Returns metric_df with updated store level optimal_location.

    Parameters:
            metric_df(DataFrame): Metrics at Store and Network Level
            store_level_new_metrics(DataFrame): updated store level optimal_location

    Returns:
            store_level_new_metrics(dict): KPIS at Store Level

    """
    metric_df.drop(columns="store_level_metrics", inplace=True)

    response = metric_df.join(
        pd.DataFrame(store_level_new_metrics, index=["store_level_metrics"]).T
    )

    return response


def get_all_baseline_metrics(queried_order_info: pd.DataFrame, metrics: list) -> dict:
    """
    Returns baseline KPIs for DH network.

    Parameters:
           queried_order_info(DataFrame): queried order from Big Query
           metrics(list) : List of column names in String Format

    Returns:
            baseline_metrics(dict) : Aggregation of weight parameters for customer order locations in DH network
    """

    total_num_customers = queried_order_info["num_customers"].sum()
    total_num_orders = queried_order_info["num_orders"].sum()
    total_num_locations = queried_order_info[["lat", "long"]].count()[0]
    total_gmv = queried_order_info["gmv"].sum()  # for future use case

    baseline_metrics = dict(
        zip(metrics, [total_num_locations, total_num_customers, total_num_orders])
    )
    return baseline_metrics


def get_unnested_data(nest: pd.DataFrame) -> pd.DataFrame:
    """
    Returns exploded data from its nest format

    Parameters:
           nest(DataFrame): nested information in Dataframe

    Returns:
            unnest(DataFrame) : exploded DataFrame

    """
    unnest = nest.explode(
        ["lat_long", "num_customers", "num_orders", "num_items", "gmv"]
    )
    unnest.reset_index(inplace=True)
    unnest = unnest.join(
        pd.DataFrame(unnest["lat_long"].to_list(), columns=["lat", "long"])
    )

    return unnest


def generate_constraint_based_info(
    queried_order_info: pd.DataFrame, polygons_based_on_driving: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns locations meeting constraints with weight parameters as nested and unnest format.

    Parameters:
           queried_order_info(DataFrame): queried order from Big Query
           polygons_based_on_driving(DataFrame) : 1 - T minutes Polygon for each store location based on
                                                  user inputs

    Returns:
            locations_within_constraints_nested(DataFrame): Order Locations within Time Constrained Polygon boundaries with
                                             list aggregated weight parameters -> num_customers,num_items,gmv..

    """

    locations_within_constraints_nested = pd.DataFrame()

    all_locations = list(map(tuple, queried_order_info[["long", "lat"]].values))

    for pred_code, row in polygons_based_on_driving.iterrows():

        list_of_locations = all_locations.copy()

        covered_polygons_df = get_covered_locations(row, list_of_locations)
        covered_locations_per_store = get_covered_locations_per_store(
            covered_polygons_df, queried_order_info, pred_code
        )
        weight_parameters_in_polygon = get_weighted_features_per_store(
            covered_locations_per_store, row
        )

        locations_within_constraints_nested = (
            locations_within_constraints_nested.append(weight_parameters_in_polygon)
        )

    locations_within_constraints_nested.reset_index(inplace=True)

    return locations_within_constraints_nested


def get_avg_drive_time_network(
    average_driving_time_for_all_stores_all_time: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns Avg Drive Times for the D'Mart Network for 1-T minutes Polygons.

    Parameters:
            average_driving_time_for_all_stores_all_time(DataFrame): Weighted Avg Time across Locations
            across Time

    Returns:
             network_driving_average_at_different_times_df(DataFrame): Network Avg Driving Time

    """
    timeline = (
        average_driving_time_for_all_stores_all_time["driving_time"]
        .drop_duplicates()
        .to_list()
    )

    network_driving_average_at_different_times = dict()

    for driving_time in timeline:
        average_driving_time_for_all_stores_a_time = (
            average_driving_time_for_all_stores_all_time[
                average_driving_time_for_all_stores_all_time["driving_time"]
                <= driving_time
            ]
        )
        network_driving_average_at_different_times[driving_time] = (
            average_driving_time_for_all_stores_a_time["driving_time"]
            * average_driving_time_for_all_stores_a_time[
                "total_customers_list_coverage"
            ]
        ).sum() / average_driving_time_for_all_stores_a_time[
            "total_customers_list_coverage"
        ].sum()

    network_driving_average_at_different_times_df = pd.DataFrame(
        network_driving_average_at_different_times, index=[0]
    ).T.rename(columns={0: "avg_drive_time"})

    return network_driving_average_at_different_times_df


def get_avg_drive_time_store(
    metric_df: pd.DataFrame, average_driving_time_for_all_stores_all_time: pd.DataFrame
) -> dict:
    """
    Returns polygon data in dict format.

    Parameters:
             metric_df(DataFrame): Metrics at Store and Network Level
             average_driving_time_for_all_stores_all_time(DataFrame): Weighted Avg Time across Locations across Time

    Returns:
              store_level_new_metrics(dict): KPIS at Store Level

    """

    store_level_new_metrics = dict()
    timeline = (
        average_driving_time_for_all_stores_all_time["driving_time"]
        .drop_duplicates()
        .to_list()
    )

    for driving_time in timeline:
        store_level_metrics_for_given_time = pd.DataFrame(
            eval(metric_df.loc[driving_time]["store_level_metrics"])
        ).set_index(["pred_lat", "pred_long"])
        store_level_updated_metrics = (
            average_driving_time_for_all_stores_all_time[
                average_driving_time_for_all_stores_all_time["driving_time"]
                == driving_time
            ][["pred_lat", "pred_long", "avg_driving_time"]]
            .set_index(["pred_lat", "pred_long"])
            .join(store_level_metrics_for_given_time)
        )
        store_level_new_metrics[driving_time] = ",".join(
            [
                "{"
                + f"\
        'lat' : {row['pred_lat']},\
        'long' : {row['pred_long']},\
        'customer_penetration' : {row['customer_penetration']},\
        'customer_penetration%' : {row['customer_penetration %']},\
        'overlap coverage%' : {row['overlap coverage %']},\
        'avg_driving_time' : {row['avg_driving_time']}\
        "
                + "}"
                for _, row in store_level_updated_metrics.reset_index().iterrows()
            ]
        )

    return store_level_new_metrics


def get_kpi_metrics(
    warehouse_polygon_df: pd.DataFrame,
    queried_order_info: pd.DataFrame,
    input_params: dict,
) -> pd.DataFrame:
    """
    Returns KPIs

    Parameters:
              warehouse_polygon_df(DataFrame) : path to read the polygon json
              queried_order_info(DataFrame): Location based information
              input_params(DataFrame): user input

    Returns:
               business_kpis(dict): KPIs at network and store level

    """

    polygons_based_on_driving = select_polygons_based_on_input_params(
        warehouse_polygon_df, input_params
    )

    constrained_locations_nested = generate_constraint_based_info(
        queried_order_info, polygons_based_on_driving
    )

    constrained_locations_unnest = get_unnested_data(constrained_locations_nested)

    columns_for_metrics = [
        "number_of_locations_covered",
        "total_customers_list_coverage",
        "total_orders_list_coverage",
    ]

    baseline_metrics = get_all_baseline_metrics(queried_order_info, columns_for_metrics)

    all_locations_for_all_time, metric_df = calculate_network_metrics(
        constrained_locations_nested, constrained_locations_unnest, baseline_metrics
    )

    average_driving_time_for_all_stores_all_time = get_weighted_driving_time(
        all_locations_for_all_time
    )[
        [
            "pred_lat",
            "pred_long",
            "driving_time",
            "total_customers_list_coverage",
            "avg_driving_time",
        ]
    ]

    network_driving_average_at_different_times_df = get_avg_drive_time_network(
        average_driving_time_for_all_stores_all_time
    )

    metric_df = metric_df.join(network_driving_average_at_different_times_df)

    store_level_new_metrics = get_avg_drive_time_store(
        metric_df, average_driving_time_for_all_stores_all_time
    )

    metric_df = update_metrics(metric_df, store_level_new_metrics)

    business_kpis = select_relevant_kpis(metric_df)

    return business_kpis


def get_network_metrics_given_time(all_locations_for_a_time):
    all_locations_for_a_time = make_coverage_data(all_locations_for_a_time)

    all_locations_for_a_time = generate_feature_metrics(all_locations_for_a_time)

    all_locations_for_a_time.reset_index(inplace=True)

    return all_locations_for_a_time


def calculate_network_metrics(
    covered_per_actual_store: pd.DataFrame,
    locations_matching_constraints: pd.DataFrame,
    baseline_metrics: dict,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Returns a set of metrics for network and store.

    Parameters:
            covered_per_actual_store(DataFrame): queried order from Big Query
            locations_matching_constraints(DataFrame) : 1 - T minutes Polygon for each store location based on
            user inputs
             baseline_metrics(dict) : Aggregation of metrics for Network

    Returns:
             all_locations_for_all_time(DataFrame): Aggregated features based on columns in optimal_location for every
                                                    location for every drive time
             metric_df(DataFrame): Metrics at Store and Network Level

    """

    all_locations_for_all_time = pd.DataFrame()
    metric_df = pd.DataFrame()
    timeline = covered_per_actual_store["driving_time"].drop_duplicates().to_list()

    for driving_time in timeline:
        locations_in_drive_time = covered_per_actual_store[
            covered_per_actual_store["driving_time"] == driving_time
        ]

        locations_unnest_in_drive_time = locations_matching_constraints[
            locations_matching_constraints["driving_time"] == driving_time
        ]

        all_locations_for_a_time = get_network_metrics_given_time(
            locations_in_drive_time
        )

        all_locations_for_a_time = get_store_overlap_given_time(
            locations_unnest_in_drive_time, all_locations_for_a_time,
            baseline_metrics
        )

        store_level_metrics_for_a_time = get_store_level_metrics(
            all_locations_for_a_time
        )

        output = get_aggregated_metrics_for_a_time(
            all_locations_for_a_time,
            baseline_metrics,
            locations_unnest_in_drive_time,
            store_level_metrics_for_a_time,
        )

        all_locations_for_all_time = all_locations_for_all_time.append(
            all_locations_for_a_time
        )

        metric_df = metric_df.append(
            pd.concat([pd.DataFrame(output, index=[driving_time])], axis=1)
        )

    return all_locations_for_all_time, metric_df


def get_store_level_metrics(all_locations_for_a_time: pd.DataFrame) -> dict:

    """
    Returns store level metrics.

    Parameters:
            all_locations_for_a_time(DataFrame): Network information for a given driving time

    Returns:
            store_level_metrics(dict): aggregated metrics at store level

    """

    store_level_metrics = (
        all_locations_for_a_time["num_customers"]
        .apply(lambda row: sum(row))
        .to_frame()
        .join(all_locations_for_a_time[["pred_lat", "pred_long", "overlap coverage %"]])
        .T.to_dict()
    )

    return store_level_metrics


def get_store_overlap_given_time(
    locations_unnest_in_drive_time: pd.DataFrame, all_locations_for_a_time: pd.DataFrame,
    baseline_metrics: dict
) -> pd.DataFrame:

    """
    Returns updated store level metrics with overlap information.

    Parameters:
        locations_unnest_in_drive_time(DataFrame) : Locations with weights for a given drive time
        all_locations_for_a_time(DataFrame): Network information for a given driving time
        baseline_metrics(dict):  Aggregation of metrics for Network

    Returns:
         all_locations_for_a_time(DataFrame): Network information for a given driving time

    """

    overlapped_per_store_for_a_time = get_customer_overlap_per_store(
        locations_unnest_in_drive_time
    )
    all_locations_for_a_time.set_index('pred_code', inplace=True)

    all_locations_for_a_time = all_locations_for_a_time.join(
        overlapped_per_store_for_a_time
    )
    all_locations_for_a_time['overlap coverage %'] = (all_locations_for_a_time['overlap_customers'] /\
                                                     baseline_metrics['total_customers_list_coverage'])*100

    all_locations_for_a_time.reset_index()

    return all_locations_for_a_time


def get_aggregated_metrics_for_a_time(
    all_locations_for_a_time,
    baseline_metrics,
    locations_unnest_in_drive_time,
    store_level_metrics_for_a_time,
):
    """
    Returns store level metrics.

    Parameters:
         all_locations_for_a_time(DataFrame): Network information for a given driving time
         baseline_metrics(dict):  Aggregation of metrics for Network
         locations_unnest_in_drive_time(DataFrame): Locations with weights for a given drive time
         store_level_metrics_for_a_time(DataFrame): aggregated metrics at store level

    Returns:
          output(dict): Aggregated metrics over a Network and Store

    """
    output = dict()

    for metric_ in list(baseline_metrics.keys()):
        sum_pred = all_locations_for_a_time[metric_].sum()
        output[f"{metric_}"] = sum_pred
        output[f"{metric_}%"] = sum_pred / baseline_metrics[metric_] * 100

    output.update(
        {
            "overlap%": (
                get_network_customer_overlap(locations_unnest_in_drive_time) / baseline_metrics["total_customers_list_coverage"]
            )
            * 100,
            "store_level_metrics": ",".join(
                [
                    "{"
                    + f"\
                                      'pred_lat' : {row['pred_lat']},\
                                      'pred_long' : {row['pred_long']},\
                                      'customer_penetration' : {row['num_customers']},\
                                      'customer_penetration %' : {(row['num_customers'] / baseline_metrics['total_customers_list_coverage']) * 100},\
                                      'overlap coverage %' : {row['overlap coverage %']}\
                                      "
                    + "}"
                    for _, row in pd.DataFrame(
                        store_level_metrics_for_a_time
                    ).T.iterrows()
                ]
            ),
        }
    )

    return output
