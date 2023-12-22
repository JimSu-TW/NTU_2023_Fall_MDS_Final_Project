import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import time
from FrontEndUtility import *
import pickle
from sklearn import preprocessing
import re

# model
model = pickle.load(open('./Model/final_random_forest.pickle', 'rb'))
utility = FrontEndUtility('./CWB_Data/', '20231221')
trainDataDf = pd.read_json('./Model/trainDataDf.json')
numCols = [col for col in trainDataDf.columns if trainDataDf[col].dtype == 'float64']
xscaler = preprocessing.StandardScaler().fit(trainDataDf[numCols])

#ui colors
colors = ['#f8b400', '#faf5e4', '#2c786c', '#004445']

progressText = "Progress data. Please wait."
loadingBar = st.progress(0, text=progressText)

# get future weather data  
def weather_data_to_df(inputData):
    timeData = []
    weatherData = []
    for entry in inputData['time']:
        timeData.append(entry['startTime'])
        weatherData.append(entry['elementValue'][0]['value'])

    # 創建DataFrame
    tempDf = pd.DataFrame({'Time': pd.to_datetime(timeData), inputData['description']: weatherData})
    return tempDf

# change wind direction str to degree
def wind_direction_str_to_degree(direction):
    directions = {
        '北風': 0,
        '偏北風': 45,
        '東北風': 45,
        '偏東風': 90,
        '東風': 90,
        '偏南風': 135,
        '東南風': 135,
        '南風': 180,
        '偏西風': 225,
        '西南風': 225,
        '西風': 270,
        '偏西北風': 315,
        '西北風': 315,
        '北風': 0,
    }
    return directions.get(direction, np.nan)

# change wind direction str to degree
def wind_speed_str_to_int(speedStr):
    numeric_part = re.search(r'\d+', speedStr).group()
    return int(numeric_part)

#get region data
basePath = "./FrontEnd_Data/"
with open(basePath + 'StationDict.json', 'r') as f:
    rStationDict = json.load(f)
    
cities = [f"{city}-{town}" for city in rStationDict for town in rStationDict[city]]
regions = {f"{city}-{town}": rStationDict[city][town] for city in rStationDict for town in rStationDict[city]}

#sidebar
selected_city = st.sidebar.selectbox('Select District:', cities)
selected_station = st.sidebar.selectbox('Select Auto Station:', regions[selected_city])
    
#initial value
utility.update_area_geo_value(selected_city, selected_station)
utility.update_area_weather_value()

#main frame data
weeklyAverageTemperature = utility.targetLocationData['weatherElement'][1]
weeklyAverageTemperatureDf = weather_data_to_df(weeklyAverageTemperature)

weeklyWindSpeed = utility.targetLocationData['weatherElement'][4]
weeklyWindSpeedDf = weather_data_to_df(weeklyWindSpeed)
tempDescription = weeklyWindSpeed['description']
weeklyWindSpeedDf[tempDescription] = weeklyWindSpeedDf[tempDescription].apply(wind_speed_str_to_int)
    
weeklyLowTemperature = utility.targetLocationData['weatherElement'][8]
weeklyLowTemperatureDf = weather_data_to_df(weeklyLowTemperature)
    
weeklyHighTemperature = utility.targetLocationData['weatherElement'][12]
weeklyHighTemperatureDf = weather_data_to_df(weeklyHighTemperature)

weeklyWindDirection = utility.targetLocationData['weatherElement'][13]
weeklyWindDirectionDf = weather_data_to_df(weeklyWindDirection)
tempDescription = weeklyWindDirection['description']
weeklyWindDirectionDf[tempDescription] = weeklyWindDirectionDf[tempDescription].apply(wind_direction_str_to_degree)

def get_turbine_status_one_hot_dict(inputStatus):
    #temp onehot
    statusOneHotdf = pd.DataFrame({'turbine_status': [f'turbine_status_{status}' for status in utility.turbineStatusList]})
    statusOneHotEncoding = pd.get_dummies(statusOneHotdf['turbine_status'])
    tempIndex = utility.turbineStatusList.index(inputStatus)
    return statusOneHotEncoding.iloc[tempIndex].to_dict()

def caculate_weekly_best_generate_power():
    weeklyData = weeklyAverageTemperatureDf.merge(weeklyWindSpeedDf, on='Time')
    weeklyData = weeklyData.merge(weeklyWindDirectionDf, on='Time')
    bestGenerateList = []
    # 定義範圍限制
    parameter_ranges = {
        'blades_angle(°)': (-180, 181),
        'gearbox_temperature(°C)': (-273, 1000),
        'engine_temperature(°C)': (0, 51),
        'motor_torque(N-m)': (-1500, 3501),
        'resistance(ohm)': (-1000, 5001),
        'blade_breadth(m)': (10, 101)
    }

    # 設定遺傳算法的參數
    population_size = 15
    generations = 25
    mutation_rate = 0.25
    loadingBarStatus = 1/len(weeklyData)
    for index, row in weeklyData.iterrows():
        loadingBar.progress(index*loadingBarStatus+loadingBarStatus, text=progressText)
        # 預測發電量
        weatherDict1 = {
            'wind_speed(m/s)': [row[weeklyWindSpeed['description']]],
            'atmospheric_temperature(°C)': [row[weeklyAverageTemperature['description']]]
        }

        windmillDict1 = {'blades_angle(°)': 0, 'gearbox_temperature(°C)': 0,
                            'engine_temperature(°C)': 0, 'motor_torque(N-m)': 0
                            }

        weatherDict2 = {
            'atmospheric_pressure(Pascal)': [101380],
            'area_temperature(°C)': [row[weeklyAverageTemperature['description']]],
            'wind_direction(°)': [row[weeklyWindDirection['description']]],
        }
        windmillDict2 = {'resistance(ohm)': 0, 'blade_breadth(m)': 0}

        statusDict = get_turbine_status_one_hot_dict('BA')

        keys = list(windmillDict1.keys())
        keys.extend(list(windmillDict2.keys()))
        baseDict = weatherDict1.copy()
        baseDict.update(windmillDict1)
        baseDict.update(weatherDict2)
        baseDict.update(windmillDict2)
        baseDict.update(statusDict)
        
        # 初始化隨機族群
        population = []
        for _ in range(population_size):
            individual = {
                param: np.random.randint(param_range[0], param_range[1]) if param != 'blade_breadth(m)' else (np.random.randint(param_range[0], param_range[1])*0.01)
                for param, param_range in parameter_ranges.items()
            }
            population.append(individual)

        # 遺傳算法的主迴圈
        for generation in range(generations):
            # 評估族群中每個個體的預測功率
            fitness_scores = []
            for individual in population:
                input_data = pd.DataFrame({**baseDict, **individual}, index=[0])
                input_data[numCols] = xscaler.transform(input_data[numCols])
                predicted_power = model.predict(input_data)[0]
                fitness_scores.append(predicted_power)

            # 找到最佳個體及其預測功率
            best_index = np.argmax(fitness_scores)
            best_individual = population[best_index]
            best_power = fitness_scores[best_index]

            # 選擇優良個體進行繁殖
            # 進行排序，得到的索引按照預測功率從大到小的順序排列
            selected_indices = np.argsort(fitness_scores)[-population_size // 2:]
            selected_population = [population[i] for i in selected_indices]

            ## 進行交叉操作（向量化）
            offspring = []
            # 確保新一代的個體數量達到族群大小 
            for _ in range(population_size - len(selected_population)):
                parent1, parent2 = np.random.choice(selected_population, size=2, replace=True)
                crossover_points = np.random.randint(0, 2, size=len(parameter_ranges), dtype=bool)
                child = {
                    param: int(parent1[param] if crossover_points[i] else parent2[param]) if param != 'blade_breadth(m)' else parent1[param] if crossover_points[i] else parent2[param]
                    for i, param in enumerate(parameter_ranges)
                }
                offspring.append(child)

            # 進行變異操作
            for child in offspring:
                for param in parameter_ranges:
                    if param != 'blade_breadth(m)' and np.random.rand() < mutation_rate:
                        # 變異產生整數
                        child[param] = np.random.randint(parameter_ranges[param][0], parameter_ranges[param][1] + 1)
                    elif param == 'blade_breadth(m)' and np.random.rand() < mutation_rate:
                        # blade_breadth 產生小數
                        child[param] = (np.random.randint(parameter_ranges[param][0], parameter_ranges[param][1])*0.01)
            population = selected_population + offspring
            
        tempDict = {'Time': row['Time'], 'Power': best_power}
        for k in keys:
            tempDict[k] = best_individual[k]
        bestGenerateList.append(tempDict)
        
    loadingBar.progress(100, text='Done!')
    bestGenerateDf = pd.DataFrame.from_records(bestGenerateList)
    weeklyData = weeklyData.merge(bestGenerateDf, on='Time')
    return weeklyData

#session initialize
if('atmospheric_pressure' not in st.session_state):
    st.session_state['atmospheric_pressure'] = utility.get_atmospheric_pressure()
if('wind_speed' not in st.session_state):
    st.session_state['wind_speed'] = utility.get_wind_speed()
if('wind_direction' not in st.session_state):
    st.session_state['wind_direction'] = utility.get_wind_direction()
if('area_temperature' not in st.session_state):
    st.session_state['area_temperature'] = utility.get_area_temperature()
if('atmospheric_temperature' not in st.session_state):
    st.session_state['atmospheric_temperature'] = utility.get_atmospheric_temperature()
if('blades_angle' not in st.session_state):
    st.session_state['blades_angle'] = utility.get_blades_angle()
if('gear_box_temperature' not in st.session_state):
    st.session_state['gear_box_temperature'] = utility.get_gear_box_temperature()
if('engine_temperature' not in st.session_state):
    st.session_state['engine_temperature'] = utility.get_engine_temperature()
if('motor_torque' not in st.session_state):
    st.session_state['motor_torque'] = utility.get_motor_torque()
if('resistance' not in st.session_state):
    st.session_state['resistance'] = utility.get_resistance()
if('turbine_status' not in st.session_state):
    st.session_state['turbine_status'] = utility.get_turbine_status()
if('blade_breadth' not in st.session_state):
    st.session_state['blade_breadth'] = utility.get_blade_breadth()
if('predict_power' not in st.session_state):
    input_data = utility.get_input_data()
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
if('weeklyPredictDataFrame' not in st.session_state):
    st.session_state['weeklyPredictDataFrame'] = caculate_weekly_best_generate_power()

if st.sidebar.button('Submit', type="primary"):
    st.session_state['atmospheric_pressure'] = utility.get_atmospheric_pressure()
    st.session_state['wind_speed'] = utility.get_wind_speed()
    st.session_state['wind_direction'] = utility.get_wind_direction()
    st.session_state['area_temperature'] = utility.get_area_temperature()
    st.session_state['atmospheric_temperature'] = utility.get_atmospheric_temperature()
    st.session_state['blades_angle'] = utility.get_blades_angle()
    st.session_state['gear_box_temperature'] = utility.get_gear_box_temperature()
    st.session_state['engine_temperature'] = utility.get_engine_temperature()
    st.session_state['motor_torque'] = utility.get_motor_torque()
    st.session_state['resistance'] = utility.get_resistance()
    st.session_state['turbine_status'] = utility.get_turbine_status()
    st.session_state['blade_breadth'] = utility.get_blade_breadth()
    input_data = utility.get_input_data()
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    st.session_state['weeklyPredictDataFrame'] = caculate_weekly_best_generate_power()
        
st.sidebar.markdown(f"<hr style='border-width: 6px'>", unsafe_allow_html=True)  
#update variables
def update_atmospheric_pressure():
    utility.set_atmospheric_pressure(st.session_state['atmospheric_pressure'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_wind_speed():
    utility.set_wind_speed(st.session_state['wind_speed'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_wind_direction():
    utility.set_wind_direction(st.session_state['wind_direction'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_area_temperature():
    utility.set_area_temperature(st.session_state['area_temperature'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_atmospheric_temperature():
    utility.set_atmospheric_temperature(st.session_state['atmospheric_temperature'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_blades_angle():
    utility.set_blades_angle(st.session_state['blades_angle'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_gear_box_temperature():
    utility.set_gear_box_temperature(st.session_state['gear_box_temperature'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_engine_temperature():
    utility.set_engine_temperature(st.session_state['engine_temperature'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_motor_torque():
    utility.set_motor_torque(st.session_state['motor_torque'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_resistance():
    utility.set_resistance(st.session_state['resistance'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_turbine_status():
    utility.set_turbine_status(st.session_state['turbine_status'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

def update_blade_breadth():
    utility.set_blade_breadth(st.session_state['blade_breadth'])
    input_data = utility.get_input_data()
    loadingBar.progress(50, text=progressText)
    input_data[numCols] = xscaler.transform(input_data[numCols])
    st.session_state['predict_power'] = np.float64(model.predict(input_data)[0]).item()
    loadingBar.progress(100, text='Done!')

# 橫拉桿輸入
st.sidebar.markdown(f"<h5 style='text-align: left; font-size: 18px; font-weight:bold'>天氣調整項</h5>", unsafe_allow_html=True)
atmospheric_pressure = st.sidebar.slider("atmospheric_pressure", -1188624, 1272551, step=1000, key="atmospheric_pressure", on_change=update_atmospheric_pressure())
wind_speed = st.sidebar.slider("wind_speed", float(0), float(700), step=0.1, key="wind_speed", on_change=update_wind_speed())
wind_direction = st.sidebar.slider("wind_direction", 0, 360, step=1, key="wind_direction", on_change=update_wind_direction())
area_temperature = st.sidebar.slider("area_temperature", float(-30), float(60), step=0.1, key="area_temperature", on_change=update_area_temperature())
atmospheric_temperature = st.sidebar.slider("atmospheric_temperature", float(-110), float(85), step=0.1, key="atmospheric_temperature", on_change=update_atmospheric_temperature())
st.sidebar.markdown(f"<hr style='border-width: 6px'>", unsafe_allow_html=True)

st.sidebar.markdown(f"<h5 style='text-align: left; font-size: 18px; font-weight:bold'>風機調整項</h5>", unsafe_allow_html=True)
blades_angle = st.sidebar.slider("Blades_angle", int(-180), int(180), step=1, key="blades_angle", on_change=update_blades_angle())
gearbox_temperature = st.sidebar.slider("gearbox_temperature", int(-273), int(999), step=1, key="gear_box_temperature", on_change=update_gear_box_temperature())
engine_temperature = st.sidebar.slider("engine_temperature", int(0), int(50), step=1, key="engine_temperature", on_change=update_engine_temperature())
motor_torque = st.sidebar.slider("motor_torque", int(-1500), int(3500), step=1, key="motor_torque", on_change=update_motor_torque())
resistance = st.sidebar.slider("resistance", int(-1000), int(5000), step=1, key="resistance", on_change=update_resistance())
blade_breadth = st.sidebar.slider("blade_breadth", float(0.1), float(1.0), step=0.01, key="blade_breadth", on_change=update_blade_breadth())
turbine_status = st.sidebar.selectbox("turbine_status", utility.turbineStatusList, key="turbine_status", on_change=update_turbine_status())

def caculte_weekly_generate_power():
    tempTimeList = []
    tempPowerList = []
    weeklyData = weeklyAverageTemperatureDf.merge(weeklyWindSpeedDf, on='Time')
    weeklyData = weeklyData.merge(weeklyWindDirectionDf, on='Time')
    for index, row in weeklyData.iterrows():
        # 預測發電量
        input_data = pd.DataFrame({
            'wind_speed(m/s)': [row[weeklyWindSpeed['description']]],
            'atmospheric_temperature(°C)': [row[weeklyAverageTemperature['description']]],
            'blades_angle(°)': [blades_angle],
            'gearbox_temperature(°C)': [gearbox_temperature],
            'engine_temperature(°C)': [engine_temperature],
            'motor_torque(N-m)': [motor_torque],
            'atmospheric_pressure(Pascal)': [atmospheric_pressure],
            'area_temperature(°C)': [row[weeklyAverageTemperature['description']]],
            'wind_direction(°)': [row[weeklyWindDirection['description']]],
            'resistance(ohm)': [resistance],
            'blade_breadth(m)': [blade_breadth],
            **get_turbine_status_one_hot_dict(turbine_status)
        })
        input_data[numCols] = xscaler.transform(input_data[numCols])
        tempPredictPower = np.float64(model.predict(input_data)[0]).item()
        
        tempTimeList.append(row['Time'])
        tempPowerList.append(tempPredictPower)
        
    return tempTimeList, tempPowerList

#main frame
st.markdown(f"<h1 style='color: {colors[3]};'>Windmill Power Prediction</h1>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: 12px; color: {colors[3]}; background-color: #ddeedf; padding: 8px; margin-top: 8px;'>Location: {selected_city}, {selected_station}</div>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: left; font-size: 30px; font-weight:bold; color: {colors[2]}'>Predicted Wind Power Generation (Kw/h)</h3>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📈 Predictive Analysis", "📃 Wind Power Dispatch Recommendations"])


with tab1:
    col1, col2 = st.columns(2)
    st.markdown(f"<h3 style='text-align: left; font-size: 30px; font-weight:bold; color: {colors[2]}'>7-Day Weather Forecast</h3>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)

    # top-left cell predict power
    with col1:
        st.markdown(f"<h3 style='text-align: left; font-size: 30px; font-weight:bold; color: {colors[2]}'>Predicted Wind Power Generation (Kw/h)</h3>", unsafe_allow_html=True)
        st.markdown(f"<span style='text-align: center; font-size:55px; font-weight:bold; color: {colors[0]}'>{st.session_state['predict_power']:.8f}</span>", unsafe_allow_html=True)

    #  top-right cell
    futureTimeList, predictPowerList= caculte_weekly_generate_power()
    df_chart = pd.DataFrame({'Date': futureTimeList, 'Power Generation': predictPowerList})
    #wait for edit
    with col2:
        st.markdown(f"<h3 style='text-align: left; font-size: 30px; font-weight:bold; color: {colors[2]}'>Weekly Wind Power Generation (Kw/h)</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=colors[1])
        ax.plot(df_chart['Date'], df_chart['Power Generation'], marker='o', color=colors[3])
        ax.set_xlabel('Date', color=colors[3])
        ax.tick_params(axis='x', colors=colors[3])
        ax.tick_params(axis='y', colors=colors[3])
        plt.box(on=None)
        ax.grid(color=colors[3], linestyle='-', linewidth=0.2, alpha=0.5)
        st.pyplot(fig)

    tempDescription = weeklyAverageTemperature['description']
    with col3:
        st.markdown(f"<h6 style='text-align: center; color: {colors[2]}'>{tempDescription}</h6>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=colors[1])
        ax.plot(weeklyAverageTemperatureDf['Time'], weeklyAverageTemperatureDf[tempDescription], marker='o', color=colors[3])
        ax.set_xlabel('Time', color=colors[3])
        ax.tick_params(axis='x', colors=colors[3])
        ax.tick_params(axis='y', colors=colors[3])
        plt.box(on=None)
        ax.grid(color=colors[3], linestyle='-', linewidth=0.2, alpha=0.5)
        st.pyplot(fig)

    tempDescription = weeklyWindSpeed['description']
    with col4:
        st.markdown(f"<h6 style='text-align: center; color: {colors[2]}'>{tempDescription}</h6>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=colors[1])
        ax.plot(weeklyWindSpeedDf['Time'], weeklyWindSpeedDf[tempDescription], marker='o', color=colors[3])
        ax.set_xlabel('Time', color=colors[3])
        ax.tick_params(axis='x', colors=colors[3])
        ax.tick_params(axis='y', colors=colors[3])
        plt.box(on=None)
        ax.grid(color=colors[3], linestyle='-', linewidth=0.2, alpha=0.5)
        st.pyplot(fig)

    tempDescription = weeklyLowTemperature['description']
    with col5:
        st.markdown(f"<h6 style='text-align: center; color: {colors[2]}'>{tempDescription}</h6>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=colors[1])
        ax.plot(weeklyLowTemperatureDf['Time'], weeklyLowTemperatureDf[tempDescription], marker='o', color=colors[3])
        ax.set_xlabel('Time', color=colors[3])
        ax.tick_params(axis='x', colors=colors[3])
        ax.tick_params(axis='y', colors=colors[3])
        plt.box(on=None)
        ax.grid(color=colors[3], linestyle='-', linewidth=0.2, alpha=0.5)
        st.pyplot(fig)

    tempDescription = weeklyHighTemperature['description']
    with col6:
        st.markdown(f"<h6 style='text-align: center; color: {colors[2]}'>{tempDescription}</h6>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=colors[1])
        ax.plot(weeklyHighTemperatureDf['Time'], weeklyHighTemperatureDf[tempDescription], marker='o', color=colors[3])
        ax.set_xlabel('Time', color=colors[3])
        ax.tick_params(axis='x', colors=colors[3])
        ax.tick_params(axis='y', colors=colors[3])
        plt.box(on=None)
        ax.grid(color=colors[3], linestyle='-', linewidth=0.2, alpha=0.5)
        st.pyplot(fig)

    #wind direction plot
    tempDescription = weeklyWindDirection['description']
    weeklyWindDirectionDf[tempDescription] = np.radians(weeklyWindDirectionDf[tempDescription])
    st.markdown(f"<h6 style='text-align: center; color: {colors[2]}'>{tempDescription}</h6>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(17, 5), facecolor=colors[1])
    ax.scatter(weeklyWindDirectionDf['Time'], [1] * len(weeklyWindDirectionDf), marker='o', color=colors[3], s=1000, zorder=1)
    ax.quiver(weeklyWindDirectionDf['Time'], [1] * len(weeklyWindDirectionDf),
            np.cos(weeklyWindDirectionDf[tempDescription]), 
            np.sin(weeklyWindDirectionDf[tempDescription]),
            scale=25, color=colors[3], zorder=2)
    ax.set_xlabel('Time', color=colors[3])
    ax.tick_params(axis='x', colors=colors[3])
    ax.set_yticks([])  # 不顯示 y 軸刻度
    plt.box(on=None)
    ax.grid(color=colors[3], linestyle='-', linewidth=0.2, alpha=0.5)
    st.pyplot(fig)

with tab2:
    st.markdown(f"""
    <div style='color: {colors[0]};'>
    ⚠️ The momentary wind force is excessive, 
    surpassing the maximum output cutout of the wind turbine.
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(st.session_state['weeklyPredictDataFrame'])
