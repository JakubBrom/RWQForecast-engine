import json
import os
import time
import openeo
import warnings
import scipy.signal
import uuid

import pandas as pd

import numpy as np
import geopandas as gpd

from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from shapely.geometry import Point

from .AIHABs_wrappers import measure_execution_time
from .get_random_points import get_sampling_points
from .get_meteo import getLastDateInDB



def authenticate_OEO():
    # Authenticate
    connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()

    return connection

def last_access(osm_id, db_name, user, db_access_date, n_days=1):
    """
    Get the last access date to CDSE for the reservoir

    :param osm_id: OSM object id
    :param db_name: Database name
    :param user: Database user
    :param db_access: Database table with access
    :return: Continue calculation (bool)
    """

    # Connect to PostGIS
    engine = create_engine('postgresql://{user}@/{db_name}'.format(user=user, db_name=db_name))
    
    # Test if the date for the osm_id exists
    query = text(f"SELECT EXISTS (SELECT 1 FROM {db_access_date} WHERE osm_id = '{osm_id}')")
    
    with engine.connect() as connection:
        result = connection.execute(query).scalar()
    
    # If the date exists, get the last access date and update the date if it is older than today - n_days   
    if result:
        query = text("SELECT MAX(date) FROM {db_table} WHERE osm_id = '{osm_id}'".format(db_table=db_access_date, osm_id=osm_id))        
        
        today = datetime.now().date()
        with engine.connect() as connection:
            last_access = connection.execute(query).scalar()
        
        print(last_access)
        
        if last_access < today - timedelta(days=n_days):
            query = text("UPDATE {db_table} SET date = '{today}' WHERE osm_id = '{osm_id}'".format(db_table=db_access_date, today=today, osm_id=osm_id))
            
            with engine.connect() as connection:
                connection.execute(query)
                connection.commit()
            
            continue_calc = True
        else:
            continue_calc = False
            
    # If the date does not exist, add the date to the table
    else:
        query = text("INSERT INTO {db_table} (osm_id, date) VALUES ('{osm_id}', '{today}')".format(db_table=db_access_date, osm_id=osm_id, today=datetime.now().date()))
        with engine.connect() as connection:
            connection.execute(query)
            connection.commit()
            
        continue_calc = True
            
    engine.dispose()        

    print(f"Continue calculation: {continue_calc}")

    return continue_calc

def process_s2_points_OEO(provider_id, client_id, client_secret, osm_id, point_layer, start_date, end_date, db_name, user, db_table, oeo_backend_url="https://openeo.dataspace.copernicus.eu", max_cc=30, cloud_mask=True):
    """
    The function processes Sentinel-2 satellite data from the Copernicus Dataspace Ecosystem. The function
    retrieves data based on the specified parameters (cloud mask) for randomly selected points within the reservoir (
    point layer). The S2 data are downloaded for defined time period. The data are stored to the PostGIS database.
    The output is a GeoDataFrame.

    Parameters:
    :param connection: OpenEO connection
    :param osm_id: OSM object id
    :param point_layer: Point layer (GeoDataFrame)
    :param start_date: Start date
    :param end_date: End date
    :param db_name: Database name
    :param user: Database user
    :param db_table: Database table
    :param max_cc: Maximum cloud cover
    :param cloud_mask: Apply cloud mask
    :return: GeoDataFrame with Sentinel-2 data for the randomly selected points for the defined time period
    """

    # Authenticate Open EO account
    connection = openeo.connect(url=oeo_backend_url)
    connection.authenticate_oidc_client_credentials(provider_id=provider_id, client_id=client_id, client_secret=client_secret)

    # Transform input GeoDataFrame layer into json
    points = json.loads(point_layer.to_json())

    # Connect to PostGIS
    engine = create_engine('postgresql://{user}@/{db_name}'.format(user=user, db_name=db_name))

    # Get bands names
    collection_info = connection.describe_collection("SENTINEL2_L2A")
    bands = collection_info['cube:dimensions']['bands']
    band_list = bands['values'][0:15]

    # Getting data
    datacube = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=[start_date, end_date],
        max_cloud_cover=max_cc,
        bands=band_list,
    )

    # Apply cloud mask etc.
    if cloud_mask:

        scl = datacube.band("SCL")
        mask = ~((scl == 6) | (scl == 2))

        # 2D gaussian kernel
        g = scipy.signal.windows.gaussian(11, std=1.6)
        kernel = np.outer(g, g)
        kernel = kernel / kernel.sum()

        # Morphological dilation of mask: convolution + threshold
        mask = mask.apply_kernel(kernel)
        mask = mask > 0.1

        datacube_masked = datacube.mask(mask)

    else:
        datacube_masked = datacube

    # Datacube aggregation
    aggregated = datacube_masked.aggregate_spatial(
        geometries=points,
        reducer="mean",
    )

    # Run the job
    job = aggregated.create_job(title=f"{osm_id}_{start_date}_{end_date}", out_format="CSV")

    # Get job ID
    jobid = job.job_id

    print(f"Job ID: {jobid}")

    # Start the job
    try:
        job.start_and_wait()

    except Exception as e:
        print(e)
        engine.dispose()
        return jobid

    # Download the results
    try:
        if job.status() == 'finished':
            # Create temporary CSV file
            csv_file = f"{uuid.uuid4()}.csv"
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp', csv_file)       # XXX: přesměrovat do tmp složky

            # Download the results
            job.get_results().download_file(csv_path)

            # Check if the file is available
            print("Waiting for data to be available...")
            t0 = time.time()
            while not os.path.exists(csv_path):
                if time.time() - t0 > 360:                  # up to 6 minutes
                    print(f"Data are not available.")
                time.sleep(1)

            df = pd.read_csv(csv_path, skip_blank_lines=True)

            print("Data has been downloaded!")
            

            print("Writing data to the PostGIS table:")
            # Convert to GeoDataFrame
            if not df.empty:
                print("DF is not empty")
                # Convert date do isoformat
                df['date'] = pd.to_datetime(df['date']).dt.date

                print("Datum ok")

                # Remove missing values
                df_all = df.dropna(axis=0, how='any')

                print("Dropna ok")

                # Rename columns
                df_all = df_all.rename(columns={'feature_index': 'PID'})
                for i in range(0, len(band_list)):
                    df_all = df_all.rename(columns={'avg(band_{})'.format(i): band_list[i]})

                print("Rename ok")

                # Add OSM id
                df_all['osm_id'] = osm_id

                print("OSM ID ok")

                # Convert to GeoDataFrame
                latlon = pd.DataFrame(point_layer['PID'])
                latlon['lat'] = point_layer.geometry.y
                latlon['lon'] = point_layer.geometry.x
                df_all = df_all.merge(latlon, on='PID', how='left')

                geometries = [Point(xy) for xy in zip(df_all['lon'], df_all['lat'])]
                gdf_out = gpd.GeoDataFrame(df_all, geometry=geometries, crs='epsg:4326')

                print("GeoDataFrame ok")

                # Save the results to the database
                gdf_out.to_postgis(db_table, con=engine, if_exists='append', index=False)
                engine.dispose()

                print(gdf_out.head())

                print("Done!")

            # Remove the temporary file
            print(f"Removing temporary file {csv_file}")
            if os.path.exists(csv_path):
                os.remove(csv_path)

            print(f"Data for OSM_ID: {osm_id} in the time window {start_date} and {end_date} has been downloaded!")
            return jobid

        else:
            print(f"Data are not available.")
            engine.dispose()
            return jobid

    except Exception as e:
        print(e)
        print(f"Data are not available.")
        engine.dispose()
        return jobid


def check_job_error(connection, jobid=None):
    """
    Check if the dataset is empty

    :param connection: OpenEO connection
    :param jobid:
    :return:
    """
    # Connection to OEO
    if not connection or connection is None:
        warnings.warn("Connection to OpenEO is not established!", stacklevel=2)
        
        return

    # Check if the error is in the log
    if jobid is not None:
        status = connection.job(jobid).status()

        if status == 'error':

            # Check if the data are available
            log = connection.job(jobid).logs()

            for i in log:

                # subs_fail = "Exception during Spark execution"
                subs_nodata = "NoDataAvailable"

                if subs_nodata in i['message']:
                    print("No data available for the time window and spatial extent")
                    return False

            # In case of unspecific error
            print("Unspecific error...")
            return True

        # In case the job was ok
        else:
            print("Dataset OK")
            return False

    # In case job_id does not exist
    else:
        print("Unspecific error... Job ID does not exist")
        return True


@measure_execution_time
def get_s2_points_OEO(provider_id, client_id, client_secret, osm_id, db_name, user, db_table_reservoirs, db_table_points, db_table_S2_points_data, db_access_date, oeo_backend_url="https://openeo.dataspace.copernicus.eu",
                       start_date=None, end_date=None, n_points_max=10000, **kwargs):
    """
    This function is a wrapper for the get_sentinel2_data function. It calls it with the defined parameters,
    manage the time windows and the database connection.

    :param connection: OpenEO connection
    :param osm_id: OSM water reservoir id
    :param db_name: Database name
    :param user: Database user
    :param db_table_reservoirs: Database table with water reservoirs
    :param db_table_points: Database table with points for reservoirs
    :param db_table_S2_points_data: Database table with Sentinel-2 data where the data are stored
    :param db_table_meteo: Database table with historic meteo data
    :param start_date: Start date
    :param end_date: End date
    :param n_points_max: Maximum number of points for water reservoir
    :param kwargs: Kwargs
    :return: None
    """

    print("Running the analysis for the reservoir with OSM ID: {osm_id}".format(osm_id=osm_id))

    # Test the last access date
    continue_calc = last_access(osm_id, db_name, user, db_access_date)
    
    if not continue_calc:
        print(f"Data for the reservoir {osm_id} are up to date.")
        return
    
    # Connect to PostGIS
    engine = create_engine('postgresql://{user}@/{db_name}'.format(user=user, db_name=db_name))

    # Get points
    point_layer = get_sampling_points(osm_id, db_name, user, db_table_reservoirs, db_table_points)

    # Check if table exists and create new one if not
    query = text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '{tab_name}')".format(
            tab_name=db_table_S2_points_data))

    with engine.connect() as connection:
        result = connection.execute(query)
        exists = result.scalar()

    # Set start date
    if exists:
        # Get last date from database
        st_date = getLastDateInDB(osm_id, db_name, user, db_table_S2_points_data)

    else:
        st_date = None

    if st_date is not None:
        st_date = st_date + timedelta(days=1)
    else:
        if start_date is None:
            start_date = '2015-06-01'

        st_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    # Set end date
    if end_date is None:
        end_date = datetime.now().date()  # Up to today
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    if st_date >= end_date:
        print('Data for period from {st_date} to {end_date} are not available. Data will not be downloaded'.format(st_date=st_date,
                                                                                            end_date=end_date))

        return

    print('Data for period from {st_date} to {end_date} will be downloaded'.format(st_date=st_date, end_date=end_date))

    # Set time windows
    # Create chunks for time series
    # Get possible length of steps (chunks)
    n_points: int = len(point_layer)

    step_length = int(n_points_max) // (n_points/100)

    n_days = (end_date - st_date).days

    n_chunks = n_days // step_length + 1

    # Create time windows
    t_delta = int((end_date - st_date).days / n_chunks)

    if t_delta < 2:
        t_delta = 2

    freq = '{tdelta}D'.format(tdelta=t_delta)
    date_range = pd.date_range(start=st_date, end=end_date, freq=freq)
    slots = [(date_range[i].date().isoformat(), (date_range[i + 1] - timedelta(days=1)).date().isoformat()) for i in
             range(len(date_range) - 1)]
    # slots.append((date_range[-1].date().isoformat(), end_date.isoformat()))       # Add last day window

    # Get Sentinel-2 data - run the job
    # Loop over the time windows
    for i in range(len(slots)):
        print(f"Data for time slot from {slots[i][0]} to {slots[i][1]} will be downloaded")

        # Try to get Sentinel-2 data for the time window. There are 2 attempts
        dataset_err = True

        # Attempt to get Sentinel-2 data
        attempt_no = 1
        while dataset_err:
            try:
                print(f"Attempt no. {attempt_no} to get Sentinel 2 data.")
                jobid = process_s2_points_OEO(provider_id, client_id, client_secret, osm_id, point_layer, slots[i][0], slots[i][1], db_name, user, db_table_S2_points_data, oeo_backend_url=oeo_backend_url)
                print(f"Job ID: {jobid}")
                dataset_err = False

            except Exception as e:
                print(f"Attempt no. {attempt_no} to get Sentinel 2 data failed. Error: {str(e)}")
                jobid = None

            # # Check if there is an error in the job
            # dataset_err = check_job_error(connection, jobid)

            if dataset_err:
                warnings.warn(f"Attempt no. {attempt_no} to get Sentinel 2 data failed.", stacklevel=2)
                time.sleep(5)
                if attempt_no == 3:
                    break
            attempt_no = attempt_no + 1

        print(f"Dataset error: {dataset_err}")

        if dataset_err:
            # Split time window to smaller windows
            warnings.warn(f"Attempt to get Sentinel 2 data failed. The time window will be split to smaller windows", stacklevel=2)
            st_in_slot = datetime.strptime(slots[i][0], "%Y-%m-%d").date()
            end_in_slot = datetime.strptime(slots[i][1], "%Y-%m-%d").date()
            n_days_window = (end_in_slot - st_in_slot).days

            # Split time window to smaller windows (30 days max)
            if n_days_window > 30:
                t_delta_window = 30
            else:
                t_delta_window = n_days_window

            freq_window = '{tdelta}D'.format(tdelta=t_delta_window)
            date_range_window = pd.date_range(start=st_in_slot, end=end_in_slot, freq=freq_window)
            slots_window = [(date_range_window[j].date().isoformat(), (date_range_window[j + 1] - timedelta(
                days=1)).date().isoformat()) for j in range(len(date_range_window) - 1)]
            slots_window.append((date_range_window[-1].date().isoformat(), end_in_slot.isoformat()))

            #
            for slot in range(len(slots_window)):
                # Attempt to download data in the time windows. If the data are not available, the attempt will be
                # repeated 5 times. In case of an error, the function will use shorter time windows as a protection
                # of the missing data. Because there can be some blocks in the server, the function is sleeping for 1
                # second between attempts.

                print(slots_window[slot][0], slots_window[slot][1])

                max_attempts = 2
                attempt = 0
                success = False

                while attempt < max_attempts and not success:
                    try:
                        process_s2_points_OEO(connection, osm_id, point_layer, slots_window[slot][0], slots_window[slot][1], db_name, user,
                                              db_table_S2_points_data)
                        success = True
                    except Exception as e:
                        warnings.warn("Attempt {attempt} failed. Error: {error}".format(attempt=attempt, error=str(e)),
                                      stacklevel=2)
                        attempt += 1
                        time.sleep(1)       # sleep for 1 second because the possibly unblocking the server

    engine.dispose()

    return
