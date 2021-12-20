

def generate_aoi(drive_time_selected, queried_order_info, locations_matching_constraints):
    """
    Returns Area of Interest

            Parameters:
                   drive_time_selected(int): Set time constraint
                   queried_order_info(DataFrame) : Order data across DH network
                   locations_matching_constraints : Existing coverage

            Returns:
                    area_of_interest(DataFrame): DH network yet to be covered

    """
    locations_covered = locations_matching_constraints[locations_matching_constraints['driving_time'] == drive_time_selected]

    area_of_interest = queried_order_info[~queried_order_info.index.isin(locations_covered.index)]

    return area_of_interest
