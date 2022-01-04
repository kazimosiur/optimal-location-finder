import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import folium
from streamlit_folium import folium_static
from folium import Marker
from folium.plugins import MarkerCluster

BUSINESS_KPIS = [
    'customer_penetration %',
    'overlap %',
    'avg_driving_time',
    'driving_time'
]


def read_kpi_csv():

    uploaded_path = '~/Desktop/metric_record.csv'
    if uploaded_path is not None :
        df = pd.read_csv(uploaded_path)
        df.rename(columns={'index': 'No_of_stores'}, inplace=True)
        return df
    else:
        return None


def get_driving_time_slider(st,df):
    driving_time_selected = st.slider(
                                    'driving_time',
                                    min_value= int(df['driving_time'].min()),
                                    max_value= int(df['driving_time'].max()),
                                    step=1
                                  )
    return driving_time_selected


def get_no_of_stores_slider(st,df):
    no_of_stores_selected = st.slider(
                                    'No_of_stores',
                                    min_value=int(df['No_of_stores'].min()),
                                    max_value=int(df['No_of_stores'].max()),
                                    step=1
                                  )
    return no_of_stores_selected


def read_warehouse_information():
    path = '~/Desktop/warehouse_SG_SIN (1).csv'
    if path is not None:
        df = pd.read_csv(path)
        return df[['latitude','longitude']]
    else:
        return None


def run_the_model():
    pass


def map_the_locations(m, df, color):

    for num, row in df.iterrows():
        folium.Marker(
            [row['latitude'], row['longitude']],
            icon=folium.Icon(color=color)
        ).add_to(m)
    return m


def main():

    df = read_kpi_csv()

    if df is not None:
        driving_time_selected = get_driving_time_slider(st, df)

        st.write("""
           Select Driving Time to Check Overlap vs Customer Penetration at Network Level
        """)
        fig, ax = plt.subplots(1, 1)

        df[df['driving_time'] == driving_time_selected].set_index('No_of_stores')[
            ['customer_penetration %', 'overlap %']
        ].plot(ax=ax, marker='o', grid=True)
        ax.set_ylabel('Percentage %')

        st.write(fig)

        st.write("""
            Select Stores to Check Business KPIs
         """)
        no_of_stores_selected = get_no_of_stores_slider(st, df)

        partitioned_df = df[
                            (df['No_of_stores'] == no_of_stores_selected)
                            &
                            (df['driving_time'] == driving_time_selected)
                            ]

        metric_to_write = partitioned_df[BUSINESS_KPIS].set_index('driving_time').copy()
        locations_to_plot = partitioned_df['list_of_locations'].values[0]

        proposed_warehouse_df = pd.DataFrame(
                                            eval(locations_to_plot),
                                            columns=['latitude', 'longitude']
                                            )
        existing_warehouse_df = read_warehouse_information()

        # Writing Metrics and Map Locations
        st.write(metric_to_write)

        st.write('Locations for opening stores')
        st.write(proposed_warehouse_df)
        # st.map(proposed_locations_df)
        location_point = [
                            existing_warehouse_df.loc[0]['latitude'],
                            existing_warehouse_df.loc[0]['longitude']
                        ]
        mapped_center = folium.Map(location=location_point,zoom_start=10)

        st.markdown("Locate the Stores on Map")
        user_choice = st.radio(
                    'Enter your choice',
                    ('Existing Warehouse', 'Proposed Warehouse', 'Both')
                )

        # add marker for existing warehouse
        if user_choice == 'Existing Warehouse':
            mapped_center = map_the_locations(mapped_center, existing_warehouse_df, color='black')
        elif user_choice == 'Proposed Warehouse':
            mapped_center = map_the_locations(mapped_center, proposed_warehouse_df, color='green')
        elif user_choice == 'Both':
            mapped_center = map_the_locations(mapped_center, existing_warehouse_df, color='black')
            mapped_center = map_the_locations(mapped_center, proposed_warehouse_df, color='green')
        else:
            pass

        # proposed_warehouse_map = map_the_locations(m, proposed_warehouse_df, color='green')

        # call to render Folium map in Streamlit
        folium_static(mapped_center)


if __name__ == '__main__':
    main()





