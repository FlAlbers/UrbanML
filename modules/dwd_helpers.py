
"""
Author: Flemming Albers

This script downloads weather data from the DWD (Deutscher Wetterdienst) API
for a specific station and time period. It then exports the data to a CSV file.

The script performs the following steps:
1. Imports necessary libraries and modules.
2. Configures the settings for the data request.
3. Defines the desired weather parameters to be downloaded.
4. Iterates over the parameters and makes a data request for each parameter.
5. Merges the parameter values into a single DataFrame.
6. Exports the merged DataFrame to a CSV file.

!!!Warning problems came up when requesting more than 5 Years of data with 5 minute resolution
!!!Therefore the time period is set to 5 years in the script.
!!!If you want to request more data, you can request the data in smaller time periods and merge the results.
"""

import pandas as pd
import wetterdienst as wd
from datetime import datetime
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import (
    DwdObservationDataset,
    DwdObservationRequest,
)

def downloadDWDByStationId(param = 'precipitation_height', resolution = 'minute_5' , station_id = 1766, start_date = "2014-01-01", end_date = "2019-01-01"):
    # Konfiguration
    settings = Settings(
        ts_shape="long",  # tidy data
        ts_humanize=True,  # humanized parameters
        ts_si_units=True  # convert values to SI units
    )

    values = pd.DataFrame()

    request = DwdObservationRequest(
        parameter=[param],
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
        settings=settings
    ).filter_by_station_id(station_id=station_id)

    stations = request.df
    paramValues = request.values.all().df
    paramValues = paramValues.rename({'value': param})
    paramValues = paramValues.to_pandas()
    if values.empty:
        values = paramValues[['date', param]]
    else:
        values = values.merge(paramValues[['date', param]], on='date', how='outer')
    paramValues = paramValues.drop(columns=param)

    # export to csv
    values.set_index('date', inplace=True)
    return values

def save_dwd_to_csv(save_path = 'precip.csv' , param = 'precipitation_height', resolution = 'minute_5' , station_id = 1766, start_date = "2014-01-01", end_date = "2019-01-01"):
    values = downloadDWDByStationId(param, resolution, station_id, start_date, end_date)
    values.to_csv(save_path , index="date")


############################################################################################################
"""
This part is only needed to transform the data to the format that SWMM can read.

This part performs the following steps:
1. Transforms the data into a format that SWMM (Storm Water Management Model) can read.
2. Converts the date column into a datetime object and separates year, month, day, hour, and minute.
3. Adds the station ID to the DataFrame.
4. Exports the transformed DataFrame to a CSV file.
"""

def save_precip_to_swmm_format(save_folder, station_id = 1766, precip_data = None):
    # convert the date column into a datetime object
    precip_data['date'] = pd.to_datetime(precip_data['date'])

    # separate the date column into year, month, day, hour, and minute
    precip_data['year'] = precip_data['date'].dt.year
    precip_data['month'] = precip_data['date'].dt.month
    precip_data['day'] = precip_data['date'].dt.day
    precip_data['hour'] = precip_data['date'].dt.hour
    precip_data['minute'] = precip_data['date'].dt.minute
    precip_data['station'] = station_id

    precip_data[['station', 'year', 'month', 'day', 'hour', 'minute', 'precipitation_height']].to_csv(f'{save_folder}\\precip_{station_id}.dat',
                                                                                                    sep=' ',
                                                                                                    index=False,
                                                                                                    header=False)


#######################################################################################################
'''
This Part prints the available parameters for a specific station, date, and resolution.

Query for available parameters for a 
- station id
- start date
- end date
- resolution
'''

def get_available_parameters(station_id = 1766, start_date = "2014-01-01", end_date = "2019-01-01", resolution = 'minute_5'):
    settings = Settings(
        ts_shape="long",  # tidy data
        ts_humanize=True,  # humanized parameters
        ts_si_units=True  # convert values to SI units
    )

    request = DwdObservationRequest(
        parameter=DwdObservationDataset,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
        settings=settings
    ).filter_by_station_id(station_id=station_id)

    params = request.values.all().df.to_pandas()

    return params.groupby('parameter').groups.keys()


if __name__ == '__main__':
    print(get_available_parameters(end_date="2015-01-01"))
    save_dwd_to_csv(end_date="2015-01-01")
    save_precip_to_swmm_format('03_sim_data\\inp', end_date="2015-01-01")
    precip = downloadDWDByStationId(end_date="2015-01-01")