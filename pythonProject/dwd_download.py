
"""
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
from wetterdienst import Settings, Resolution, Period
from wetterdienst.provider.dwd.observation import (
    DwdObservationDataset,
    DwdObservationPeriod,
    DwdObservationRequest,
    DwdObservationResolution,
    DwdObservationParameter,
    download
)

def download_weather_data():
    # Konfiguration
    settings = Settings(
        ts_shape="long",  # tidy data
        ts_humanize=True,  # humanized parameters
        ts_si_units=True  # convert values to SI units
    )

    # Datenanfrage
    dwd_dataset = DwdObservationDataset
    attrib = [
        'precipitation_height',
        # 'temperature_air_mean_200',
        # 'wind_speed',
    ]

    values = pd.DataFrame()

    for value in attrib:
        request = DwdObservationRequest(
            parameter=[value],
            resolution="minute_5",
            start_date="2014-01-01",
            end_date="2019-01-01",
            settings=settings
        ).filter_by_station_id(station_id=1766)

        stations = request.df
        paramValues = request.values.all().df
        paramValues = paramValues.rename({'value': value})
        paramValues = paramValues.to_pandas()
        if values.empty:
            values = paramValues[['date', value]]
        else:
            values = values.merge(paramValues[['date', value]], on='date', how='outer')
        paramValues = paramValues.drop(columns=value)

    # export to csv
    values.set_index('date', inplace=True)
    values.to_csv('pythonProject\\P_FMO2.csv', index="date")


    ############################################################################################################
    """
    This part is only needed to transform the data to the format that SWMM can read.

    This part performs the following steps:
    1. Transforms the data into a format that SWMM (Storm Water Management Model) can read.
    2. Converts the date column into a datetime object and separates year, month, day, hour, and minute.
    3. Adds the station ID to the DataFrame.
    4. Exports the transformed DataFrame to a CSV file.
    """

    
    station_id=1766
    valuesSWMM = values
    # convert the date column into a datetime object
    valuesSWMM['date'] = pd.to_datetime(valuesSWMM['date'])

    # separate the date column into year, month, day, hour, and minute
    valuesSWMM['year'] = valuesSWMM['date'].dt.year
    valuesSWMM['month'] = valuesSWMM['date'].dt.month
    valuesSWMM['day'] = valuesSWMM['date'].dt.day
    valuesSWMM['hour'] = valuesSWMM['date'].dt.hour
    valuesSWMM['minute'] = valuesSWMM['date'].dt.minute
    valuesSWMM['station'] = station_id

    valuesSWMM[['station', 'year', 'month', 'day', 'hour', 'minute', 'precipitation_height']].to_csv('klima_1766.dat',
                                                                                                      sep=' ',
                                                                                                      index=False,
                                                                                                      header=False)



#######################################################################################################
    '''
    This Part prints the available parameters for a specific station, date, and resolution.

    Query for available parameters for a 
     - specific station 
     - date 
     - resolution
    '''

    request = DwdObservationRequest(
        parameter=DwdObservationDataset,
        resolution="hourly",
        start_date="1996-01-01",
        end_date="1996-01-01",
        settings=settings
    ).filter_by_station_id(station_id=1766)  # enter station id here

    params = request.values.all().df.to_pandas()

    params.groupby('parameter').groups.keys()

