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

# Konfiguration
settings = Settings( # default
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
        resolution= "minute_5",
        start_date="2014-01-01",
        end_date="2019-01-01",
        settings=settings
    ).filter_by_station_id(station_id=1766)

    stations = request.df
    # stations
    paramValues = request.values.all().df
    paramValues = paramValues.rename({'value': value})
    #print(paramValues)
    paramValues = paramValues.to_pandas()
    #print(paramValues.groupby('parameter').groups.keys())
    if values.empty:
        values = paramValues[['date', value]]
    else:
        values = values.merge(paramValues[['date', value]], on='date', how='outer')
    paramValues = paramValues.drop(columns=value)


# export to csv
values.set_index('date', inplace=True)
values.to_csv('pythonProject\\P_FMO2.csv',index="date")



############################################################################################################
# convert the date column into a datetime object
valuesSWMM = values
valuesSWMM['date'] = pd.to_datetime(valuesSWMM['date'])

valuesSWMM['year'] = valuesSWMM['date'].dt.year
valuesSWMM['month'] = valuesSWMM['date'].dt.month
valuesSWMM['day'] = valuesSWMM['date'].dt.day
valuesSWMM['hour'] = valuesSWMM['date'].dt.hour
valuesSWMM['minute'] = valuesSWMM['date'].dt.minute
valuesSWMM['station'] = station_id


# valuesSWMM['time'] = valuesSWMM['date'].dt.time
# # valuesSWMM['date'] = valuesSWMM['date'].dt.date

# valuesSWMM['date'] = valuesSWMM['date'].dt.strftime('%m/%d/%Y')
# valuesSWMM['time'] = valuesSWMM['time'].astype(str).str[:5]
# show the modified data frame
# valuesSWMM
# valuesSWMM = valuesSWMM.set_index(['date','time'])
valuesSWMM



valuesSWMM[['station','year','month','day','hour','minute','precipitation_height']].to_csv('klima_1766.dat', sep=' ',index=False, header=False)


#**Print available parameters**
# Query for available parameters for a 
# - specific station 
# - date 
# - resolution


request = DwdObservationRequest(
        parameter=DwdObservationDataset,
        resolution= "hourly",
        start_date="1996-01-01",
        end_date="1996-01-01",
        settings=settings
    ).filter_by_station_id(station_id=1766) #enter station id here

params = request.values.all().df.to_pandas()

params.groupby('parameter').groups.keys()