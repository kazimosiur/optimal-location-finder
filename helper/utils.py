import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from collections import Counter
import itertools
from datetime import datetime
from typing import Union, Tuple


def read_json(path_to_file: str) -> dict:
    """
    Returns polygon data in dict format.

            Parameters:
                   path_to_file (str): sql query

            Returns:
                    json_(dict): 1 - T minutes Polygon for each store location
    """
    json_ = pd.read_json(path_to_file).to_dict()
    return json_


def get_drive_modes(json_: dict) -> set:
    """
    Get drive modes from json.
            Parameters:
                   json_(dict): 1 - T minutes Polygon for each store location

            Returns:
                    driving_modes(set): set of drive modes in string
    """

    driving_modes = set()
    for key_detail in json_.keys():
        driving_modes.add(key_detail.split('_')[-1:][0])
    return driving_modes


def label_drive_modes(driving_modes: set) -> dict:
    """
    Get labelled drive modes.
            Parameters:
                   driving_modes(dict): set of drive modes in string

            Returns:
                    driving_mode_maps(dict): key -> drive modes value -> label
    """
    driving_mode_maps = dict()
    for label, mode in enumerate(driving_modes):
        driving_mode_maps[mode] = label
    return driving_mode_maps


def invert_drive_mode_labels(driving_mode_maps):
    """
    Invert key value pairs
            Parameters:
                   driving_mode_maps(dict): key -> drive modes value -> label

            Returns:
                    drive_maps(dict): inverted key:value pairs
    """
    drive_maps = {value: key for key, value in driving_mode_maps.items()}
    return drive_maps


def convert_json_to_dataframe(json_: dict) -> pd.DataFrame:
    """
     Returns polygons based on Input Parameters in DataFrame format .

             Parameters:
                    json_(dict): Contains polygons for locations based on 1-T minutes Drive Time

             Returns:
                     warehouse_polygon_df(DataFrame): converted dict in DataFrame format
     """

    driving_modes = get_drive_modes(json_)
    driving_mode_maps = label_drive_modes(driving_modes)
    drive_maps = invert_drive_mode_labels(driving_mode_maps)

    warehouse_drivemode = json_.keys()
    warehouse_locations = list()
    for warehouse in warehouse_drivemode:
        location, drive_time, drive_mode = warehouse.split('_')
        drive_value = driving_mode_maps[drive_mode]
        polygons = json_[warehouse]
        location = location.replace('(', '').replace(')', '')
        warehouse_locations.append(f'({location}, {drive_time}, {drive_value}, {polygons})')

    warehouse_polygon_df = pd.DataFrame(eval(','.join(warehouse_locations)),
                                        columns=['latitude', 'longitude', 'driving_time', 'driving_mode', 'polygon'])

    warehouse_polygon_df['driving_mode'] = warehouse_polygon_df['driving_mode'].apply(lambda row: drive_maps[row])
    return warehouse_polygon_df


def points_inside_polygon(geo_json_poly, list_of_points=None):
    """
     Returns polygons based on Input Parameters in DataFrame format .

             Parameters:
                    geo_json_poly(Polygon): Contains polygons for locations based on 1-T minutes Drive Time
                    list_of_points(list) : list of locations

             Returns:
                     list_of_points(list): converted dict in DataFrame format
     """
    polygon = Polygon([tuple(location) for location in geo_json_poly['coordinates'][0]])
    geometries = []
    for idx, point in enumerate(list_of_points):
        point_obj = Point(point)  # create point
        list_of_points[idx] = (polygon.contains(point_obj), point)  # check if polygon contains point
        point_geo = {"type": "Point",
                     "coordinates": point
                     },
        geometries.append(point_geo)
    geometries.append(geo_json_poly)
    return list_of_points


def generate_features_per_store(weight_parameters_in_polygon: pd.DataFrame) -> pd.DataFrame:
    """
     Returns weight parameters for every location.

             Parameters:
                    weight_parameters_in_polygon(DataFrame): Contains Features as Weights associated with each location

             Returns:
                     weighted_collection_in_polygon(DataFrame): Collection of weighted parameters for every location
     """
    weight_parameters_in_polygon['lat_long'] = weight_parameters_in_polygon[['lat', 'long']].apply(lambda row: (
        row.lat,
        row.long
    ), axis=1
                                                                                                   )

    weighted_collection_in_polygon = weight_parameters_in_polygon.groupby('pred_code')['lat_long'].apply(
        list).to_frame() \
        .join(
        weight_parameters_in_polygon.groupby('pred_code')['num_customers'].apply(list).to_frame()
    ).join(
        weight_parameters_in_polygon.groupby('pred_code')['num_orders'].apply(list).to_frame()
    ).join(
        weight_parameters_in_polygon.groupby('pred_code')['num_items'].apply(list).to_frame()
    ).join(
        weight_parameters_in_polygon.groupby('pred_code')['gmv'].apply(list).to_frame()
    )
    return weighted_collection_in_polygon


def update_store_locations(lat: float, long: float, weight_parameters_in_polygon: pd.DataFrame) -> pd.DataFrame:
    """
     Returns weight parameters for every location.

             Parameters:
                 lat(float): Latitude
                 long(float): Longitude
                 weight_parameters_in_polygon(DataFrame): Contains Features as Weights associated with each location

             Returns:
                  weight_parameters_in_polygon(DataFrame): Updating DataFrame with the locations
     """
    weight_parameters_in_polygon['pred_lat'] = lat
    weight_parameters_in_polygon['pred_long'] = long
    return weight_parameters_in_polygon


def get_covered_locations_per_store(covered_polygons_df: pd.DataFrame,
                                    queried_order_info: pd.DataFrame,
                                    pred_code: int) -> pd.DataFrame:
    """
     Returns weight parameters for every location.

             Parameters:
                 covered_polygons_df(DataFrame): Locations covered by Polygons
                 queried_order_info(DataFrame): Order Locations with weight parameters
                 pred_code(int): index for a Location

             Returns:
                  weight_parameters_in_polygon(DataFrame): Updated covered polygons with locations covered and weight
                                                           associated with each location
     """

    # Adding a key for join and for aggregating weight parameters in a list
    covered_polygons_df['pred_code'] = pred_code

    covered_polygons_df.set_index(['lat', 'long'], inplace=True)

    weight_parameters_in_polygon = covered_polygons_df.join(queried_order_info.set_index(['lat', 'long'])).reset_index()

    return weight_parameters_in_polygon


def get_covered_locations(row: pd.DataFrame, list_of_locations: list) -> pd.DataFrame:
    """
     Returns weight parameters for every location.

             Parameters:
                 row(DataFrame): Locations covered by Polygons
                 list_of_locations(DataFrame): Order Locations with weight parameters

             Returns:
                  weight_parameters_in_polygon(DataFrame): Updated covered polygons with locations covered and weight
                                                           associated with each location
     """

    geo_json_poly = row['polygon']
    saved_df = pd.DataFrame(
        points_inside_polygon(geo_json_poly, list_of_locations)
    ).copy()
    covered_polygons_df = saved_df[saved_df[0]]

    covered_polygons_df = pd.DataFrame(covered_polygons_df[1].to_list(), columns=['long', 'lat'])
    covered_polygons_df['driving_time'] = row['driving_time']
    covered_polygons_df['pred_lat'] = row['latitude']
    covered_polygons_df['pred_long'] = row['longitude']

    return covered_polygons_df


def get_current_time() -> str:
    """
     Returns current timestamp.

             Returns:
                    timestamp(str): timestamp in YY-MM-DD-HR-MIN
     """

    current_timestamp = datetime.now()
    day_ = current_timestamp.day
    month_ = current_timestamp.month
    year_ = current_timestamp.year
    hour_ = current_timestamp.time().hour
    minutes_ = current_timestamp.time().minute

    timestamp = f'{year_}-{month_}-{day_}_{hour_}-{minutes_}'
    return timestamp


def get_weighted_features_per_store(weight_parameters_in_polygon: pd.DataFrame,
                                    row: pd.DataFrame) -> pd.DataFrame:
    """
     Returns weight parameters for every location.

             Parameters:
                 weight_parameters_in_polygon(DataFrame): Covered polygons with locations covered and weight
                                                           associated with each location

                 row(DataFrame): Locations covered by Polygons

             Returns:
                  weight_parameters_in_polygon(DataFrame): weight_parameters_in_polygon updated with drivetime
     """
    lat, long = row[['latitude', 'longitude']]
    weight_parameters_in_polygon = generate_features_per_store(weight_parameters_in_polygon)
    weight_parameters_in_polygon = update_store_locations(lat, long, weight_parameters_in_polygon)
    weight_parameters_in_polygon['driving_time'] = row['driving_time']

    return weight_parameters_in_polygon


def calculate_overlapping_per_store(locations_matching_constraints: pd.DataFrame) -> pd.DataFrame:
    """
     Returns weight parameters for every location.

             Parameters:
                 locations_matching_constraints(DataFrame):  Locations matching constraints

             Returns:
                  count_overlap(DataFrame): overlap optimal_location at store level
     """

    overlapped_df = locations_matching_constraints.groupby(['lat', 'long'])['pred_code'].apply(list).reset_index()[
        'pred_code'].to_frame()
    overlapped_df['length'] = overlapped_df['pred_code'].apply(len)

    count_overlap = locations_matching_constraints.groupby('pred_code')['pred_code'].count().to_frame()
    count_overlap.index.name = ''

    all_codes = list(itertools.chain.from_iterable(
        overlapped_df[overlapped_df['length'] > 1]['pred_code'].to_list()
    ))

    count_overlap = count_overlap.join(
        pd.DataFrame(Counter(all_codes), index=['overlap_count']).T,

    )
    count_overlap['overlap coverage %'] = (count_overlap['overlap_count'] / count_overlap['pred_code']) * 100

    count_overlap['overlap coverage %'] = count_overlap['overlap coverage %'].fillna(0)
    count_overlap.reset_index(inplace=True)

    count_overlap.rename(columns={'': 'pred_label'}, inplace=True)

    return count_overlap[['pred_label', 'overlap coverage %']]


def get_latlong(locs: Tuple) -> Union[list, list]:
    """
     Returns splitteed list of lat,long from locations.

             Parameters:
                 locs (DataFrame):  Locations in tuple format

             Returns:
                lat = list of latitude from locations
                long = list of longitude from locations
     """

    lat = []
    long = []
    for loc in locs:
        latitude, longitude = loc
        lat.append(np.float(latitude))
        long.append(np.float(longitude))
    return lat, long


def generate_feature_metrics(covered_per_actual_store: pd.DataFrame) -> pd.DataFrame:
    """
     Returns list of lat,long split from locations.

             Parameters:
                 covered_per_actual_store (DataFrame):  Locations matching constraints along with weight parameters

             Returns:
                 covered_per_actual_store (DataFrame):  Updated input dataframe with aggregated features
     """

    feature_columns = ['filtered_loc', 'customers_list', 'orders_list']

    covered_per_actual_store['number_of_locations_covered'] = covered_per_actual_store[feature_columns[0]].apply(
        lambda locs: len(locs))

    for feature_column in feature_columns[1:]:
        covered_per_actual_store[f'total_{feature_column}_coverage'] = covered_per_actual_store[feature_column].apply(
            lambda locs: 0 if len(locs) == 0 else sum(locs))

    return covered_per_actual_store


def make_coverage_data(covered_per_actual_store):

    """
     Returns list of lat,long split from locations.

             Parameters:
                 covered_per_actual_store (DataFrame):  Locations matching constraints along with weight parameters

             Returns:
                 covered_per_actual_store (DataFrame):  Updated input dataframe with aggregated weights in list
     """

    covered_per_actual_store['filtered_loc'] = get_filtered_locations(covered_per_actual_store)

    # add the changes here

    covered_per_actual_store['latlong_list'] = covered_per_actual_store['filtered_loc'].apply(
        lambda rows: split_list_into_features(rows)[0])
    covered_per_actual_store['customers_list'] = covered_per_actual_store['filtered_loc'].apply(
        lambda rows: split_list_into_features(rows)[1])
    covered_per_actual_store['orders_list'] = covered_per_actual_store['filtered_loc'].apply(
        lambda rows: split_list_into_features(rows)[2])

    #
    covered_per_actual_store['number_of_locations_covered'] = covered_per_actual_store['filtered_loc'].apply(
        lambda locs: len(locs))

    covered_per_actual_store['latlong_list'] = covered_per_actual_store['latlong_list'].apply(
        lambda locs: get_latlong(locs))

    return covered_per_actual_store


def get_filtered_locations(covered_per_actual_store: pd.DataFrame) -> list:
    """
     Returns Unique Locations covered by each location.

             Parameters:
                 covered_per_actual_store (DataFrame):  Locations matching constraints along with weight parameters

             Returns:
                 filtered_loc (list):  Unique Locations covered
     """
    traversed_loc = set()
    filtered_loc = []

    for num, row in covered_per_actual_store.iterrows():
        new_loc = [
            (loc, cust, orders) for loc, cust, orders in zip(
                row[f'lat_long'],
                row['num_customers'],
                row['num_orders']
            ) if loc not in traversed_loc
        ]
        traversed_loc.update(row[f'lat_long'])
        filtered_loc.append(new_loc)

    return filtered_loc


def split_list_into_features(rows: list) -> Union[list, list, list]:
    """
     Returns a collection of lists

             Parameters:
                 rows (list): list of tuples

             Returns:
                 loc_(list): list of Locations
                 num_customers(list): list of customers
                 num_orders(list):  list of orders
     """

    loc_, num_customers, num_orders = [], [], []
    for row in rows:
        lat_long, customers, orders = row[0], row[1], row[2]
        loc_.append(lat_long)
        num_customers.append(customers)
        num_orders.append(orders)
    return loc_, num_customers, num_orders


def get_weighted_driving_time(source_for_kpi_metrics: pd.DataFrame) -> pd.DataFrame:
    """
     Returns polygon data in dict format.

             Parameters:
                    source_for_kpi_metrics(DataFrame): Aggregated features based on columns selected for every
                                                            location for every drive time

             Returns:
                     all_stores(DataFrame): Aggregating features based on columns in optimal_location for every
                                                            location for every drive time

     """
    all_stores = pd.DataFrame()
    for _, row in source_for_kpi_metrics[['pred_lat', 'pred_long']].drop_duplicates().iterrows():
        one_store = source_for_kpi_metrics[(source_for_kpi_metrics['pred_lat'] == row['pred_lat']) & (
                    source_for_kpi_metrics['pred_long'] == row['pred_long'])]
        one_store['filtered_loc'] = get_filtered_locations(one_store)
        one_store['total_customers'] = one_store['filtered_loc'].apply(
            lambda rows: sum(split_list_into_features(rows)[1]))

        timeline = one_store['driving_time'].to_list()
        avg_driving_time = dict()
        for driving_time in timeline:
            selected_polygons = one_store[one_store['driving_time'] <= driving_time]
            avg_driving_time[driving_time] = (selected_polygons['driving_time'] * selected_polygons[
                'total_customers']).sum() / selected_polygons['total_customers'].sum()

        one_store['avg_driving_time'] = avg_driving_time.values()

        all_stores = all_stores.append(one_store)
    return all_stores


def calculate_avg_driving_time_at_store_level(metric_df: pd.DataFrame,
                                              average_driving_time_for_all_stores_all_time: pd.DataFrame) -> dict:
    """
     Returns polygon data in dict format.

             Parameters:
                    metric_df(DataFrame): Aggregated optimal_location over Network
             Returns:
                     store_level_new_metrics(DataFrame): AVG Driving Time at Store level

    """

    store_level_new_metrics = dict()
    timeline = average_driving_time_for_all_stores_all_time['driving_time'].drop_duplicates().to_list()

    for driving_time in timeline:

        store_level_metrics_for_given_time = pd.DataFrame(eval(metric_df.loc[driving_time]['store_level_metrics'])).\
                                                set_index(['pred_lat', 'pred_long'])

        store_level_updated_metrics = \
            average_driving_time_for_all_stores_all_time[
            average_driving_time_for_all_stores_all_time['driving_time'] == driving_time
            ][['pred_lat', 'pred_long', 'avg_driving_time']].set_index(['pred_lat', 'pred_long']).\
            join(store_level_metrics_for_given_time)

        store_level_new_metrics[driving_time] = ','.join(['{' + f"\
                                                            'pred_lat' : {row['pred_lat']},\
                                                            'pred_long' : {row['pred_long']},\
                                                            'customer_penetration' : {row['customer_penetration']},\
                                                            'customer_penetration %' : {row['customer_penetration %']},\
                                                            'overlap coverage %' : {row['overlap coverage %']},\
                                                            'avg_driving_time' : {row['avg_driving_time']}\
                                                            " + '}' for _, row in store_level_updated_metrics.reset_index().iterrows()])
    return store_level_new_metrics


def select_relevant_kpis(metric_df: pd.DataFrame) -> pd.DataFrame:
    """
     Returns KPIs renamed in DataFrame format.

             Parameters:
                    metric_df(DataFrame) : Metrics at Store and Network Level

             Returns:
                     business_kpis(DataFrame) : KPIs Business is interested in
     """
    BUSINESS_METRICS = [
        'customer_penetration_absolute',
        'customer_penetration %',
        'overlap %',
        'avg_drive_time',
        'store_level_metrics'
    ]
    METRICS_TO_LOOK = [
        'total_customers_list_coverage[pred]',
        'total_customers_list_coverage[pred] %',
        'overlap %[pred]',
        'avg_drive_time',
        'store_level_metrics'
    ]

    rename_columns = dict(zip(METRICS_TO_LOOK, BUSINESS_METRICS))
    rename_columns

    metric_df.rename(columns=rename_columns, inplace=True)

    business_kpis = metric_df[BUSINESS_METRICS]

    return business_kpis


def get_polygons_for_locations(path_for_polygon: str) -> dict:
    """
     Returns polygons for locations in dict format .

             Parameters:
                    path_for_polygon (str): path provided for json

             Returns:
                     polygons_based_on_driving(DataFrame): 1 - T minutes Polygon for each store location based on
                                                           input_params
     """

    json_ = read_json(path_for_polygon)
    return json_
