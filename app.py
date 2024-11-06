from flask import Flask, g, render_template, jsonify, request
import json
from sqlalchemy import create_engine
import config
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
import requests

app = Flask(__name__, static_url_path='')
app.config.from_object('config')


def connect_to_database():
    engine = create_engine(
        ("mysql+mysqldb://{}:{}@{}:{}/{}").format(config.USER, config.PASSWORD, config.URL, config.PORT, config.DB), echo=False)

    connection = engine.connect()
    return connection
    # return engine


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = connect_to_database()
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route("/")
def root():
    engine = get_db()
    return render_template('index.html', apikey=config.APIKEY, station_num=int(engine.execute("SELECT count(*) from station;").scalar()))


@app.route("/stations")
def stations():
    engine = get_db()
    stations = []
    rows = engine.execute("SELECT * from station;")
    for row in rows:
        stations.append(dict(row))
    return jsonify(stations)


@app.route("/stations/<int:station_id>")
def station(station_id):
    engine = get_db()
    data = []
    rows = engine.execute(
        "SELECT * from station where number = {};".format(station_id))
    for row in rows:
        data.append(dict(row))
    return jsonify(data)


@app.route("/occupancy/<int:station_id>")
def occupancy(station_id):
    engine = get_db()
    df = pd.read_sql_query(
        'select * from availability where number = %(number)s', engine, params={"number": station_id})
    df['last_update_date'] = pd.to_datetime(df.last_update, unit='s')
    df.set_index('last_update_date', inplace=True)
    res = df['available_bike_stands'].resample("1d").mean()
    print(res)
    return jsonify(data=json.dumps([(x.isoformat(), y) for x, y in zip(res.index, res.values)]))




# Load the trained model from the file
# with open('new_model.pkl', 'rb') as handle:
#     model = pickle.load(handle)

@app.route('/predictions')
def process_data():
    response = requests.get("https://api.openweathermap.org/data/2.5/forecast?q=Dublin&appid=2b9a975c8bf478cfdd6f5235ed9e235e&units=metric")
    json_data = response.json()


    dt_txt = []
    temp = []
    wind_speed = []
    humidity = []
    visibility = []
    hour_of_day = []
    week_of_day = []
    main_weather = []
    number = []
    engine = get_db()
    s_df = pd.read_sql_table('station', engine, columns=['number'])
    number = s_df['number'].tolist()
    station_df = pd.DataFrame({'number': number})

    for weather in json_data['list']:
        dt_txt_str = weather['dt_txt']
        dt_txt_obj = datetime.strptime(dt_txt_str, '%Y-%m-%d %H:%M:%S')
        hour_of_day.append(dt_txt_obj.hour)
        week_of_day.append(dt_txt_obj.strftime('%A'))
        dt_txt.append(weather['dt_txt'])
        temp.append(weather['main']['temp'])
        wind_speed.append(weather['wind']['speed'])
        humidity.append(weather['main']['humidity'])
        visibility.append(weather['visibility'])
        main_weather.append(weather['weather'][0]['main'])
    # Create a dataframe from the lists
    data = {
        'dt_txt': dt_txt,
        'temp': temp,
        'wind_speed': wind_speed,
        'humidity': humidity,
        'visibility': visibility,
        'main_weather': main_weather,
        'hour_of_day': hour_of_day,
        'week_of_day': week_of_day
    }
    weather_df = pd.DataFrame(data)


    # Convert the week_of_day column to a set of dummy variables
    week_of_day_dummies = pd.get_dummies(weather_df['week_of_day'])
    weather_df = pd.concat([weather_df, week_of_day_dummies], axis=1)

    weather_df = weather_df.drop([ 'week_of_day'],axis=1)
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Iterate through the days of the week
    for day in days_of_week:
        if day not in weather_df.columns:
            weather_df[day] = 0

    weather_df.loc[weather_df['main_weather'] == 'Drizzle', 'main_weather'] = 'Rain'
    weather_df['is_rainy'] = weather_df['main_weather'].apply(lambda x: 1 if 'rain' in x.lower() else 0)

    # weather_df = weather_df.drop(['dt_txt', 'main_weather', 'week_of_day'],axis=1)  # Drop unnecessary columns

    station_df.rename(columns={'number': 'station_id'}, inplace=True)
    station_df
    weather_df['number'] = None

    df = pd.DataFrame()

    # Loop through each row in station_df
    for index, row in station_df.iterrows():
        # Get the "station_id" value
        station_id = row['station_id']
        # Duplicate the weather data for the current station_id
        duplicated_weather_data = weather_df.copy()
        # Add a new column "number" with the current station_id value
        duplicated_weather_data['number'] = station_id
        # Append the duplicated weather data to the df DataFrame
        df = df.append(duplicated_weather_data, ignore_index=True)
        
    X = df[['hour_of_day','temp','wind_speed','humidity','visibility','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'is_rainy', 'number']]
    with open('new_model.pkl', 'rb') as handle:
        model = pickle.load(handle)
        predictions = model.predict(X)
        df['predicted_values'] = predictions
    df[df['number'] == 1]
    with open('model_bikeStands.pkl', 'rb') as handle:
        model_bikeStands = pickle.load(handle)
    X = df[['hour_of_day','temp','wind_speed','humidity','visibility','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'is_rainy', 'number']]
    prediction_bikeStand = model_bikeStands.predict(X)
    df['prediction_bikeStand'] = prediction_bikeStand
    selected_columns = ["dt_txt", "prediction_bikeStand", "predicted_values", "number"]
    df = df[selected_columns]
    df['datetime'] = pd.to_datetime(df['dt_txt'], format='%Y-%m-%d %H:%M:%S')
    # Create "date" column with formatted datetime string
    df['date'] = df['datetime'].dt.strftime('%m-%d')
    df['hour'] = df['datetime'].dt.strftime('%H:%M')
    selected_columns = ["date",'hour', "prediction_bikeStand", "predicted_values", "number"]
    df = df[selected_columns]
    grouped_data = df.groupby('number').apply(lambda x: x.to_dict(orient='records'))
    # Convert the grouped data to JSON format
    json_data = grouped_data.to_json(orient='index')
    return json_data

if __name__ == "__main__":
    app.run(debug=True)
