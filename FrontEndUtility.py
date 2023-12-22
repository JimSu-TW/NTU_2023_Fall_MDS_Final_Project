# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
import requests

class FrontEndUtility:
    def __init__(self, dataPath, dateStr):
        self.dataPath = dataPath
        self.autoStationData = None
        self.newTaipeiData = None
        self.taipeiData = None
        self.targetStationData = None
        self.targetLocationData = None
        
        #weather param
        self.atmosphericPressure = 0
        self.windSpeed = 0
        self.windDirection = 0
        self.areaTemperature = 0
        self.atmosphericTemperature = 0
        
        self.read_api_data(dateStr)
        self.predictPower = 0
        
        #windmill param
        self.bladesAngle = -9
        self.gearBoxTemperature = 41
        self.engineTemperature = 43
        self.motorTorque = 1710
        self.resistance = 1580
        self.turbineStatusList = ['A2', 'AAA', 'AB', 'ABC', 'AC', 'B', 'B2', 'BA', 'BB', 'BBB', 'BCB', 'BD', 'D']
        self.turbineStatus = 'BA'
        self.bladeBreadth = 0.4
    
    def set_atmospheric_pressure(self, input):
        self.atmosphericPressure = input
        
    def get_atmospheric_pressure(self):
        return int(self.atmosphericPressure)
    
    def set_wind_speed(self, input):
        self.windSpeed = input
    
    def get_wind_speed(self):
        return float(self.windSpeed)
    
    def set_wind_direction(self, input):
        self.windDirection = input
        
    def get_wind_direction(self):
        return int(self.windDirection)
    
    def set_area_temperature(self, input):
        self.areaTemperature = input
    
    def get_area_temperature(self):
        return float(self.areaTemperature)
    
    def set_atmospheric_temperature(self, input):  
        self.atmosphericTemperature = input
    
    def get_atmospheric_temperature(self):
        return float(self.atmosphericTemperature)
    
    def set_blades_angle(self, input):
        self.bladesAngle = input
        
    def get_blades_angle(self):
        return int(self.bladesAngle)
    
    def set_gear_box_temperature(self, input):
        self.gearBoxTemperature = input
        
    def get_gear_box_temperature(self):
        return int(self.gearBoxTemperature)
    
    def set_engine_temperature(self, input):
        self.engineTemperature = input
        
    def get_engine_temperature(self):
        return int(self.engineTemperature)
    
    def set_motor_torque(self, input):
        self.motorTorque = input
    
    def get_motor_torque(self):
        return int(self.motorTorque)
    
    def set_resistance(self, input):
        self.resistance = input
        
        
    def get_resistance(self):    
        return int(self.resistance)
    
    def set_turbine_status(self, input):
        self.turbineStatus = input
        
    def get_turbine_status(self):
        return self.turbineStatus
    
    def get_turbine_status_one_hot_dict(self):
        #temp onehot
        statusOneHotdf = pd.DataFrame({'turbine_status': [f'turbine_status_{status}' for status in self.turbineStatusList]})
        statusOneHotEncoding = pd.get_dummies(statusOneHotdf['turbine_status'])
        tempIndex = self.turbineStatusList.index(self.turbineStatus)
        return statusOneHotEncoding.iloc[tempIndex].to_dict()

    def set_blade_breadth(self, input):
        self.bladeBreadth = input
    
    def get_blade_breadth(self):
        return float(self.bladeBreadth)
    
    def get_predict_power(self):
        return self.predictPower

    def get_all_data(self):
        return {
            'atmosphericPressure': self.atmosphericPressure,
            'windSpeed': self.windSpeed,
            'windDirection': self.windDirection,
            'areaTemperature': self.areaTemperature,
            'atmosphericTemperature': self.atmosphericTemperature,
            'bladesAngle': self.bladesAngle,
            'gearBoxTemperature': self.gearBoxTemperature,
            'engineTemperature': self.engineTemperature,
            'motorTorque': self.motorTorque,
            'resistance': self.resistance,
            'turbineStatus': self.turbineStatus,
            'bladeBreadth': self.bladeBreadth,
            'predictPower': self.predictPower
        }
    
    def read_api_data(self, dateStr):
        with open(self.dataPath + dateStr + '_AutoData.json', 'r') as f:
            self.autoStationData = json.load(f)

        with open(self.dataPath + dateStr +'_NewTaipei7Day.json', 'r') as f:
            self.newTaipeiData = json.load(f)

        with open(self.dataPath + dateStr +'_Taipei7Day.json', 'r') as f:
            self.taipeiData = json.load(f)
    
    def set_auto_station_data(self, stationName):
        stationDataList = self.autoStationData['records']['Station']
        for stationData in stationDataList:
            if stationData['StationName'] == stationName:
                self.targetStationData = stationData
                break
            
    def set_weekly_predict_data(self, selected_city):
        city = selected_city.split('-')[0]
        town = selected_city.split('-')[1]
        
        if city == '臺北市':
            locationDataList = self.taipeiData['records']['locations'][0]['location']
        else:
            locationDataList = self.newTaipeiData['records']['locations'][0]['location']
            
        for index in range(len(locationDataList)):
            if locationDataList[index]['locationName'] == town:
                self.targetLocationData = locationDataList[index]
                break
    
    def update_area_geo_value(self, selected_city, stationName):
        self.set_weekly_predict_data(selected_city)
        self.set_auto_station_data(stationName)
    
    #update api value
    def update_area_weather_value(self):
        self.atmosphericPressure = self.targetStationData['WeatherElement']['AirPressure']*100
        self.windSpeed = self.targetStationData['WeatherElement']['WindSpeed']
        self.windDirection = self.targetStationData['WeatherElement']['WindDirection']
        self.areaTemperature = self.targetStationData['WeatherElement']['AirTemperature']
        self.atmosphericTemperature = self.targetLocationData['weatherElement'][1]['time'][0]['elementValue'][0]['value']
    
    def get_input_data(self):
        # 預測發電量
        input_data = pd.DataFrame({
            'wind_speed(m/s)': [self.windSpeed],
            'atmospheric_temperature(°C)': [self.atmosphericTemperature],
            'blades_angle(°)': [self.bladesAngle],
            'gearbox_temperature(°C)': [self.gearBoxTemperature],
            'engine_temperature(°C)': [self.engineTemperature],
            'motor_torque(N-m)': [self.motorTorque],
            'atmospheric_pressure(Pascal)': [self.atmosphericPressure],
            'area_temperature(°C)': [self.areaTemperature],
            'wind_direction(°)': [self.windDirection],
            'resistance(ohm)': [self.resistance],
            'blade_breadth(m)': [self.bladeBreadth],
            **self.get_turbine_status_one_hot_dict()
        })
        return input_data
                
                
