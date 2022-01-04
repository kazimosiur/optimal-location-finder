from .metrics_kpi import *


def generate_aoi(
    queried_order_info: pd.DataFrame, locations_matching_constraints: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns Area of Interest

    Parameters:
           queried_order_info(DataFrame) : Order data across DH network
           locations_matching_constraints : Existing coverage meeting constraints

    Returns:
            area_of_interest(DataFrame): DH network yet to be covered

    """
    queried_order_info["lat_long"] = list(
        zip(queried_order_info["lat"], queried_order_info["long"])
    )
    queried_order_info.set_index("lat_long", inplace=True)
    locations_matching_constraints.set_index("lat_long", inplace=True)
    area_of_interest = queried_order_info[
        ~queried_order_info.index.isin(locations_matching_constraints.index)
    ]

    return area_of_interest


def get_existing_coverage(
    input_params: dict,
    warehouse_polygon_df: pd.DataFrame,
    queried_order_info: pd.DataFrame,
) -> pd.DataFrame:

    """
    Returns existing coverage

    Parameters:
           input_params(int) : User input
           warehouse_polygon_df(DataFrame) : Polygons for locations for 1-T minutes Driving Time
           queried_order_info(DataFrame): Customer order locations

    Returns:
            constrained_locations_unnest(DataFrame): Existing coverage meeting constraints

    """

    polygons_based_on_driving = select_polygons_based_on_input_params(
        warehouse_polygon_df, input_params
    )

    polygons_within_time = polygons_based_on_driving[
        polygons_based_on_driving["driving_time"] == input_params["driving_time"]
    ]

    constrained_locations_nested = generate_constraint_based_info(
        queried_order_info, polygons_within_time
    )

    constrained_locations_unnest = get_unnested_data(constrained_locations_nested)

    return constrained_locations_unnest
