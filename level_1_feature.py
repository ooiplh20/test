# Create Windows
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def read_and_process_location_data(file_path):
    """Read and initial process of location data"""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['DateTime', 'LocationCode', 'Power(mW)'])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['LocationCode'] = df['LocationCode'].astype(int)
    return df

def check_window_validity(group_data, date):
    """Check window validity based on competition criteria"""
    yesterday = date - timedelta(days=1)

    # Check yesterday's data (entire day)
    yesterday_data = group_data[group_data['DateTime'].dt.date == yesterday]

    # Check today's early data (7:00-9:00)
    today_early_data = group_data[
        (group_data['DateTime'].dt.date == date) &
        (group_data['DateTime'].dt.hour.between(7, 8))
        ]

    return (
            yesterday_data['Power(mW)'].max() > 0 and
            today_early_data['Power(mW)'].max() > 0
    )

def create_id(dt, location_code):
    """Create competition format ID: YYYYMMDDHHMMLL"""
    return int(f"{dt.strftime('%Y%m%d%H%M')}{location_code:02d}")

def process_all_data(data_path):
    """Process all locations and create windows"""
    all_windows = []
    all_window_labels = {}  # Dictionary to store labels for each window

    for loc_num in range(1, 18):
        file_path = data_path / f'L{loc_num}_Train_cleaned.csv'
        if not file_path.exists():
            continue

        print(f"Processing location {loc_num}...")
        loc_df = read_and_process_location_data(file_path)
        unique_dates = sorted(loc_df['DateTime'].dt.date.unique())

        for date in unique_dates[1:]:  # Skip first date as we need yesterday's data
            if check_window_validity(loc_df, date):
                yesterday = date - timedelta(days=1)

                # Get window data (full previous day + current day until 9:00)
                window_data = loc_df[
                    (loc_df['DateTime'].dt.date == yesterday) |
                    ((loc_df['DateTime'].dt.date == date) &
                     (loc_df['DateTime'].dt.hour < 9))
                    ].copy()

                # Get label data
                label_data = loc_df[
                    (loc_df['DateTime'].dt.date == date) &
                    (loc_df['DateTime'].dt.hour.between(9, 16))
                    ].copy()

                if len(window_data) > 0:
                    # Create unique window identifier (just for internal use)
                    window_key = f"{date.strftime('%Y%m%d')}{loc_num:02d}"

                    # Add idx to window data
                    window_data['idx'] = window_data.apply(
                        lambda x: create_id(x['DateTime'], x['LocationCode']),
                        axis=1
                    )

                    all_windows.append(window_data)

                    if len(label_data) == 48:  # Only store if we have complete labels
                        # Create minimal label data with only idx and Power(mW)
                        label_data['idx'] = label_data.apply(
                            lambda x: create_id(x['DateTime'], x['LocationCode']),
                            axis=1
                        )
                        all_window_labels[window_key] = label_data[['idx', 'Power(mW)']]

    # Create combined labels DataFrame with only idx and Power(mW)
    train_labels = pd.concat(all_window_labels.values()) if all_window_labels else pd.DataFrame(columns=['idx', 'Power(mW)'])

    return all_windows, train_labels, set(all_window_labels.keys())

def split_windows(windows, train_window_keys, submission_path):
    """Split windows into train and test sets"""
    # Read submission template
    submission_df = pd.read_csv(submission_path)
    submission_ids = set(submission_df['序號'].astype(str))

    train_windows = []
    test_windows = []

    for window in windows:
        # Get date and location from first row
        first_row = window.iloc[0]
        date = first_row['DateTime'].date() + timedelta(days=1)  # Next day for prediction
        loc_num = first_row['LocationCode']
        window_key = f"{date.strftime('%Y%m%d')}{loc_num:02d}"

        # Check if this window is in the submission template
        test_ids = [f"{date.strftime('%Y%m%d')}{hour:02d}{minute:02d}{loc_num:02d}"
                    for hour in range(9, 17)
                    for minute in range(0, 60, 10)]

        is_test = any(str(test_id) in submission_ids for test_id in test_ids)

        if is_test:
            test_windows.append(window)
        elif window_key in train_window_keys:
            train_windows.append(window)

    # Verify all submission IDs are covered
    covered_ids = set()
    for window in test_windows:
        first_row = window.iloc[0]
        date = first_row['DateTime'].date() + timedelta(days=1)
        loc_num = first_row['LocationCode']

        for hour in range(9, 17):
            for minute in range(0, 60, 10):
                id_str = f"{date.strftime('%Y%m%d')}{hour:02d}{minute:02d}{loc_num:02d}"
                covered_ids.add(id_str)

    missing_ids = submission_ids - covered_ids
    if missing_ids:
        print(f"Warning: {len(missing_ids)} submission IDs not covered in test windows!")
        print("First few missing IDs:", sorted(list(missing_ids))[:5])

    return train_windows, test_windows

def main():
    data_path = Path('cleaned_data')
    submission_path = Path('36_TestSet_SubmissionTemplate/upload.csv')

    print("Processing all data...")
    windows, train_labels, train_window_keys = process_all_data(data_path)

    print("Splitting into train and test sets...")
    train_windows, test_windows = split_windows(windows, train_window_keys, submission_path)

    print(f"\nDataset Statistics:")
    print(f"Number of training windows: {len(train_windows)}")
    print(f"Number of testing windows: {len(test_windows)}")
    print(f"Number of training labels: {len(train_labels)}")

    # Sample check of first window
    if train_windows:
        print("\nFirst training window example:")
        print("Columns:", train_windows[0].columns.tolist())
        print("Time range:", train_windows[0]['DateTime'].min(), "to", train_windows[0]['DateTime'].max())

    return train_windows, test_windows, train_labels

if __name__ == "__main__":
    train_windows, test_windows, train_labels = main()

## Weather Features
 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_weather_data():
    """Load and preprocess weather data"""
    # Load hourly data
    hour_data = pd.read_csv('hualien_weather_hour.csv')
    hour_data['obsTime'] = pd.to_datetime(hour_data['obsTime'])
    hour_data = hour_data[hour_data['station_id'] == 'C0Z100']

    # Load minute data
    min_data = pd.read_csv('hualien_weather_min.csv')
    min_data['datetime'] = pd.to_datetime(min_data['datetime'])

    return hour_data, min_data

def calculate_minute_features(window_data, weather_min, target_time):
    """Calculate minute-level features for a specific target time"""
    # Get relevant weather data
    time_range = pd.date_range(
        target_time - timedelta(minutes=20),
        target_time + timedelta(minutes=20),
        freq='10T'
    )

    relevant_weather = weather_min[weather_min['datetime'].isin(time_range)].copy()
    if len(relevant_weather) == 0:
        return pd.Series()

    # Calculate current, previous and next values
    current_idx = relevant_weather['datetime'].searchsorted(target_time)

    features = {}

    # Basic minute-level features
    for col in ['wind_speed', 'temperature', 'humidity', 'pressure', 'precipitation',
                'max_gust_speed', 'max_gust_direction', 'uv_index']:
        if current_idx > 0 and current_idx < len(relevant_weather):
            features[f'min_{col}'] = relevant_weather.iloc[current_idx][col]
            features[f'min_{col}_prev10'] = relevant_weather.iloc[current_idx-1][col]
            features[f'min_{col}_next10'] = relevant_weather.iloc[current_idx+1][col]
            features[f'min_{col}_diff'] = features[f'min_{col}'] - features[f'min_{col}_prev10']

    # Window statistics (30min, 1h, 2h)
    window_ranges = {
        '30min': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2)
    }

    for window_name, delta in window_ranges.items():
        window_weather = weather_min[
            (weather_min['datetime'] >= target_time - delta) &
            (weather_min['datetime'] < target_time)
            ]

        if len(window_weather) > 3:  # Minimum number of points for meaningful statistics
            for col in ['wind_speed', 'temperature', 'humidity', 'pressure', 'precipitation']:
                prefix = f'min_{col}_{window_name}'
                features[f'{prefix}_mean'] = window_weather[col].mean()
                features[f'{prefix}_std'] = window_weather[col].std()
                features[f'{prefix}_sum'] = window_weather[col].sum()
                try:
                    features[f'{prefix}_kurt'] = float(window_weather[col].kurt())  # Explicitly convert to float
                except:
                    features[f'{prefix}_kurt'] = 0.0  # Default value if kurtosis calculation fails
                features[f'{prefix}_peak'] = window_weather[col].max()

    return pd.Series(features)

def calculate_hourly_features(window_data, weather_hour, target_time):
    """Calculate hour-level features for a specific target time"""
    # Get target hour time (rounded down to hour)
    target_hour = target_time.floor('H')

    # Get relevant weather data for ±2 hours
    relevant_weather = weather_hour[
        (weather_hour['obsTime'] >= target_hour - timedelta(hours=2)) &
        (weather_hour['obsTime'] <= target_hour + timedelta(hours=2))
        ].copy()

    features = {}

    if len(relevant_weather) == 0:
        return pd.Series(features)

    # Basic hourly features
    hour_cols = {
        'WDIR': 'wind_direction',
        'WDSD': 'wind_speed',
        'TEMP': 'temperature',
        'HUMD': 'humidity',
        'PRES': 'pressure'
    }

    # Find the current hour's data
    current_data = relevant_weather[relevant_weather['obsTime'] == target_hour]
    prev1_data = relevant_weather[relevant_weather['obsTime'] == target_hour - timedelta(hours=1)]
    prev2_data = relevant_weather[relevant_weather['obsTime'] == target_hour - timedelta(hours=2)]

    # Calculate basic features for non-precipitation columns
    for col, name in hour_cols.items():
        try:
            if len(current_data) > 0:
                features[f'hour_{name}'] = float(current_data[col].iloc[0])
            if len(prev1_data) > 0:
                features[f'hour_{name}_prev1'] = float(prev1_data[col].iloc[0])
            if len(prev2_data) > 0:
                features[f'hour_{name}_prev2'] = float(prev2_data[col].iloc[0])

            # Calculate diff only if we have both current and prev1
            if len(current_data) > 0 and len(prev1_data) > 0:
                features[f'hour_{name}_diff'] = features[f'hour_{name}'] - features[f'hour_{name}_prev1']

        except (IndexError, ValueError) as e:
            features[f'hour_{name}'] = np.nan
            features[f'hour_{name}_prev1'] = np.nan
            features[f'hour_{name}_prev2'] = np.nan
            features[f'hour_{name}_diff'] = np.nan

    # Handle precipitation separately
    try:
        # Convert '.' to 0 for precipitation data
        relevant_weather['H_24R'] = relevant_weather['H_24R'].replace('.', '0').astype(float)

        if len(current_data) > 0:
            features['hour_precipitation'] = float(current_data['H_24R'].iloc[0])
        if len(prev1_data) > 0:
            features['hour_precipitation_prev1'] = float(prev1_data['H_24R'].iloc[0])
        if len(prev2_data) > 0:
            features['hour_precipitation_prev2'] = float(prev2_data['H_24R'].iloc[0])

        if len(current_data) > 0 and len(prev1_data) > 0:
            features['hour_precipitation_diff'] = features['hour_precipitation'] - features['hour_precipitation_prev1']

    except (IndexError, ValueError) as e:
        features['hour_precipitation'] = 0.0
        features['hour_precipitation_prev1'] = 0.0
        features['hour_precipitation_prev2'] = 0.0
        features['hour_precipitation_diff'] = 0.0

    # Window statistics (3h, 6h)
    window_ranges = {
        '3h': timedelta(hours=3),
        '6h': timedelta(hours=6)
    }

    for window_name, delta in window_ranges.items():
        window_weather = weather_hour[
            (weather_hour['obsTime'] >= target_hour - delta) &
            (weather_hour['obsTime'] <= target_hour)
            ]

        if len(window_weather) >= 3:
            # Calculate stats for non-precipitation columns
            for col, name in hour_cols.items():
                prefix = f'hour_{name}_{window_name}'
                try:
                    series = pd.to_numeric(window_weather[col].replace('.', np.nan))

                    features[f'{prefix}_mean'] = float(series.mean())
                    features[f'{prefix}_std'] = float(series.std())
                    features[f'{prefix}_sum'] = float(series.sum())
                    if len(series.dropna()) > 3:
                        features[f'{prefix}_kurt'] = float(series.kurt())
                    else:
                        features[f'{prefix}_kurt'] = 0.0
                    features[f'{prefix}_peak'] = float(series.max())
                except Exception as e:
                    features[f'{prefix}_mean'] = np.nan
                    features[f'{prefix}_std'] = np.nan
                    features[f'{prefix}_sum'] = np.nan
                    features[f'{prefix}_kurt'] = 0.0
                    features[f'{prefix}_peak'] = np.nan

            # Handle precipitation statistics separately
            try:
                precip_series = window_weather['H_24R'].replace('.', '0').astype(float)
                prefix = f'hour_precipitation_{window_name}'

                features[f'{prefix}_mean'] = float(precip_series.mean())
                features[f'{prefix}_std'] = float(precip_series.std())
                features[f'{prefix}_sum'] = float(precip_series.sum())
                if len(precip_series) > 3:
                    features[f'{prefix}_kurt'] = float(precip_series.kurt())
                else:
                    features[f'{prefix}_kurt'] = 0.0
                features[f'{prefix}_peak'] = float(precip_series.max())
            except Exception as e:
                features[f'{prefix}_mean'] = 0.0
                features[f'{prefix}_std'] = 0.0
                features[f'{prefix}_sum'] = 0.0
                features[f'{prefix}_kurt'] = 0.0
                features[f'{prefix}_peak'] = 0.0

    return pd.Series(features)

def generate_local_climate_features(window_data, weather_hour, weather_min):
    """Generate all local climate features for the prediction window"""
    # Get location and date information from window
    location_code = window_data['LocationCode'].iloc[0]
    prediction_date = window_data['DateTime'].iloc[-1].date()

    # Generate features for each prediction time point
    all_features = []

    # Generate prediction times (9:00-17:00, 10-minute intervals)
    prediction_times = pd.date_range(
        start=f"{prediction_date} 09:00:00",
        end=f"{prediction_date} 16:50:00",
        freq='10T'
    )

    for target_time in prediction_times:
        # Create feature row
        features = {}
        features['idx'] = int(f"{target_time.strftime('%Y%m%d%H%M')}{location_code:02d}")

        # Add minute-level features
        minute_features = calculate_minute_features(window_data, weather_min, target_time)
        features.update(minute_features)

        # Add hourly features
        hourly_features = calculate_hourly_features(window_data, weather_hour, target_time)
        features.update(hourly_features)

        all_features.append(features)

    # Create DataFrame with all features
    feature_df = pd.DataFrame(all_features)

    # Ensure idx is the first column
    cols = ['idx'] + [col for col in feature_df.columns if col != 'idx']
    feature_df = feature_df[cols]

    # drop columns where col_name has 'hour_precipitation_6h'
    feature_df = feature_df.loc[:, ~feature_df.columns.str.contains('hour_precipitation')]

    return feature_df

# Load weather data once (outside the function)
weather_hour = pd.read_csv('hualien_weather_hour.csv')
weather_hour['obsTime'] = pd.to_datetime(weather_hour['obsTime'])
weather_hour = weather_hour[weather_hour['station_id'] == 'C0Z100']

weather_min = pd.read_csv('hualien_weather_min.csv')
weather_min['datetime'] = pd.to_datetime(weather_min['datetime'])

generate_local_climate_features(train_windows[2], weather_hour, weather_min)

## Historical Features
 
def calculate_statistics(data, cols):
    """Calculate comprehensive statistics for given columns"""
    stats = {}

    for col in cols:
        series = data[col]
        prefix = col.replace('(', '').replace(')', '').replace('/', '_')

        # Basic statistics
        stats[f'{prefix}_mean'] = series.mean()
        stats[f'{prefix}_std'] = series.std()
        stats[f'{prefix}_sum'] = series.sum()
        stats[f'{prefix}_peak'] = series.max()

        # Kurtosis with error handling
        try:
            stats[f'{prefix}_kurt'] = float(series.kurt())
        except:
            stats[f'{prefix}_kurt'] = 0.0

        # Trend statistics
        if len(series) > 1:
            # Calculate slope using linear regression
            x = np.arange(len(series))
            slope, _ = np.polyfit(x, series, 1)
            stats[f'{prefix}_slope'] = slope

            # Calculate momentum (rate of change)
            momentum = series.diff().mean()
            stats[f'{prefix}_momentum'] = momentum

            # Calculate acceleration (rate of change of momentum)
            acceleration = series.diff().diff().mean()
            stats[f'{prefix}_acceleration'] = acceleration
        else:
            stats[f'{prefix}_slope'] = 0.0
            stats[f'{prefix}_momentum'] = 0.0
            stats[f'{prefix}_acceleration'] = 0.0

    return stats

def generate_historical_features(window_data):
    """Generate historical microclimatic and power features"""
    target_cols = [
        'WindSpeed(m/s)',
        'Sunlight(Lux)',
        'Temperature(°C)',
        'Humidity(%)',
        'Pressure(hpa)',
        'Power(mW)'
    ]

    features_list = []

    # Get basic time information
    prediction_date = window_data['DateTime'].iloc[-1].date()
    location_code = window_data['LocationCode'].iloc[0]

    # Generate prediction times
    prediction_times = pd.date_range(
        start=f"{prediction_date} 09:00:00",
        end=f"{prediction_date} 16:50:00",
        freq='10T'
    )

    for target_time in prediction_times:
        features = {}
        features['idx'] = int(f"{target_time.strftime('%Y%m%d%H%M')}{location_code:02d}")

        # 1. Morning data (7:00-9:00 today)
        morning_data = window_data[
            (window_data['DateTime'].dt.date == prediction_date) &
            (window_data['DateTime'].dt.hour.between(7, 8))
            ]
        morning_stats = calculate_statistics(morning_data, target_cols)
        features.update({f'morning_{k}': v for k, v in morning_stats.items()})

        # 2. Yesterday's data
        yesterday = prediction_date - timedelta(days=1)
        yesterday_data = window_data[window_data['DateTime'].dt.date == yesterday]

        # Yesterday morning (7:00-9:00)
        yesterday_morning = yesterday_data[
            yesterday_data['DateTime'].dt.hour.between(7, 8)
        ]
        yesterday_morning_stats = calculate_statistics(yesterday_morning, target_cols)
        features.update({f'yesterday_morning_{k}': v for k, v in yesterday_morning_stats.items()})

        # Yesterday full day
        yesterday_full_stats = calculate_statistics(yesterday_data, target_cols)
        features.update({f'yesterday_full_{k}': v for k, v in yesterday_full_stats.items()})

        # Yesterday same time
        target_hour = target_time.hour
        target_minute = target_time.minute
        yesterday_same_time = yesterday_data[
            (yesterday_data['DateTime'].dt.hour == target_hour) &
            (yesterday_data['DateTime'].dt.minute == target_minute)
            ]

        # If we have yesterday's data for the same time
        if len(yesterday_same_time) > 0:
            for col in target_cols:
                col_name = col.replace('(', '').replace(')', '').replace('/', '_')
                features[f'yesterday_same_time_{col_name}'] = yesterday_same_time[col].iloc[0]
        else:
            # If no exact match, use interpolation or nearby values
            time_window = timedelta(minutes=10)
            nearby_data = yesterday_data[
                (yesterday_data['DateTime'].dt.hour == target_hour) &
                (yesterday_data['DateTime'].dt.minute.between(target_minute - 10, target_minute + 10))
                ]

            if len(nearby_data) > 0:
                for col in target_cols:
                    col_name = col.replace('(', '').replace(')', '').replace('/', '_')
                    features[f'yesterday_same_time_{col_name}'] = nearby_data[col].mean()
            else:
                for col in target_cols:
                    col_name = col.replace('(', '').replace(')', '').replace('/', '_')
                    features[f'yesterday_same_time_{col_name}'] = np.nan

        features_list.append(features)

    # Create DataFrame and ensure proper column order
    feature_df = pd.DataFrame(features_list)
    cols = ['idx'] + [col for col in feature_df.columns if col != 'idx']
    feature_df = feature_df[cols]

    return feature_df

generate_historical_features(train_windows[0])
## Solar Features
 
import pvlib
import numpy as np
from datetime import datetime, timedelta

location_elevations = {
    1: 120,  # Top floor (5th floor)
    2: 120,
    3: 120,
    4: 120,
    5: 120,
    6: 120,
    7: 120,
    8: 60,   # 3rd floor
    9: 60,
    10: 0,   # Ground floor
    11: 0,
    12: 0,
    13: 120, # Top floor management building
    14: 120,
    15: 10,  # Ground floor Meilun
    16: 10,
    17: 20,  # Private warehouse
}

# Location coordinates dictionary
location_coordinates = {
    1: (23.8994, 121.5442),  # 23°53'58"N 121°32'40"E
    2: (23.8997, 121.5447),  # 23°53'59"N 121°32'41"E
    3: (23.8997, 121.5450),  # 23°53'59"N 121°32'42"E
    4: (23.8994, 121.5442),  # 23°53'58"N 121°32'40"E
    5: (23.8994, 121.5447),  # 23°53'58"N 121°32'41"E
    6: (23.8994, 121.5442),  # 23°53'58"N 121°32'40"E
    7: (23.8994, 121.5442),  # 23°53'58"N 121°32'40"E
    8: (23.8997, 121.5450),  # 23°53'59"N 121°32'42"E
    9: (23.8994, 121.5442),  # 23°53'58"N 121°32'40"E
    10: (23.8994, 121.5442), # 23°53'58"N 121°32'40"E
    11: (23.8997, 121.5447), # 23°53'59"N 121°32'41"E
    12: (23.8997, 121.5447), # 23°53'59"N 121°32'41"E
    13: (23.8978, 121.5394), # 23°53'52"N 121°32'22"E
    14: (23.8978, 121.5394), # 23°53'52"N 121°32'22"E
    15: (24.0092, 121.6172), # 24°0'33"N 121°37'02"E
    16: (24.0089, 121.6172), # 24°0'32"N 121°37'02"E
    17: (23.9833, 121.6000), # Approximate for private location
}

location_panel_azimuths = {
    1: 181,   # 181° South
    2: 175,   # 175° South
    3: 180,   # 180° South
    4: 161,   # 161° South
    5: 208,   # 208° Southwest
    6: 208,   # 208° Southwest
    7: 172,   # 172° South
    8: 219,   # 219° Southwest
    9: 151,   # 151° Southeast
    10: 223,  # 223° Southwest
    11: 131,  # 131° Southeast
    12: 298,  # 298° Northwest
    13: 249,  # 249° West
    14: 197,  # 197° South
    15: 127,  # 127° Southeast
    16: 82,   # 82° East
    17: 180,  # Default to South for private location
}

import pvlib
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import pytz

def get_solar_position(time, latitude, longitude, altitude):
    """Calculate solar position parameters"""
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        tz='Asia/Taipei'
    )

    # Get solar position
    solar_position = location.get_solarposition(time)

    return solar_position

def calculate_panel_solar_angle(surface_tilt, surface_azimuth, solar_position):
    """Calculate angle of incidence on panel"""
    aoi = pvlib.irradiance.aoi(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth']
    )
    return aoi

def calculate_clear_sky(location, times):
    """Calculate clear sky radiation components"""
    site = pvlib.location.Location(
        latitude=location[0],
        longitude=location[1],
        altitude=location[2],
        tz='Asia/Taipei'
    )

    clear_sky = site.get_clearsky(times, model='ineichen')
    return clear_sky

def calculate_sun_times(latitude, longitude, altitude, date):
    """Calculate sunrise, sunset, and transit times"""
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        tz='Asia/Taipei'
    )

    # Convert date to timezone-aware datetime index
    tz = pytz.timezone('Asia/Taipei')
    times = pd.DatetimeIndex([datetime.combine(date, datetime.min.time())]).tz_localize(tz)

    # Calculate sun times
    sun_times = location.get_sun_rise_set_transit(times)
    return sun_times

def generate_solar_features(window_data):
    """Generate all solar-related features"""
    location_code = window_data['LocationCode'].iloc[0]
    prediction_date = window_data['DateTime'].iloc[-1].date()

    # Get location information
    latitude, longitude = location_coordinates[location_code]
    altitude = location_elevations[location_code]
    panel_azimuth = location_panel_azimuths[location_code]
    surface_tilt = 20  # Assuming standard tilt angle

    features_list = []

    # Generate prediction times
    prediction_times = pd.date_range(
        start=f"{prediction_date} 09:00:00",
        end=f"{prediction_date} 16:50:00",
        freq='10T',
        tz='Asia/Taipei'
    )

    # Calculate daily sun times
    sun_times = calculate_sun_times(latitude, longitude, altitude, prediction_date)
    sunrise = sun_times['sunrise'].iloc[0]
    sunset = sun_times['sunset'].iloc[0]
    solar_noon = sun_times['transit'].iloc[0]
    daylight_duration = (sunset - sunrise).total_seconds() / 3600  # in hours

    # Calculate clear sky radiation for the whole day
    clear_sky = calculate_clear_sky(
        (latitude, longitude, altitude),
        prediction_times
    )

    # Calculate solar position for all times at once
    solar_position = get_solar_position(prediction_times, latitude, longitude, altitude)

    # Calculate panel solar angles
    panel_angles = calculate_panel_solar_angle(surface_tilt, panel_azimuth, solar_position)

    for i, target_time in enumerate(prediction_times):
        features = {}

        # Create ID
        features['idx'] = int(f"{target_time.strftime('%Y%m%d%H%M')}{location_code:02d}")

        # Solar position features
        features['solar_elevation'] = float(solar_position['elevation'].iloc[i])
        features['solar_azimuth'] = float(solar_position['azimuth'].iloc[i])
        features['solar_zenith'] = float(solar_position['zenith'].iloc[i])
        features['panel_solar_angle'] = float(panel_angles.iloc[i])

        # Daylight features
        features['daylight_duration'] = float(daylight_duration)
        features['time_to_solar_noon'] = float(
            (target_time - solar_noon).total_seconds() / 3600  # in hours
        )

        # Clear sky radiation features
        features['clear_sky_ghi'] = float(clear_sky['ghi'].iloc[i])
        features['clear_sky_dni'] = float(clear_sky['dni'].iloc[i])
        features['clear_sky_dhi'] = float(clear_sky['dhi'].iloc[i])

        # Air mass features
        features['relative_airmass'] = float(pvlib.atmosphere.get_relative_airmass(
            solar_position['zenith'].iloc[i]
        ))
        features['absolute_airmass'] = float(features['relative_airmass'] * (
                window_data['Pressure(hpa)'].mean() / 1013.25
        ))

        # Panel efficiency features
        temp_coeff = -0.004  # Typical temperature coefficient
        ref_temp = 25  # Reference temperature
        current_temp = window_data['Temperature(°C)'].mean()
        features['panel_efficiency_factor'] = float(1 + temp_coeff * (current_temp - ref_temp))

        # Elevation adjusted radiation
        pressure_ratio = pvlib.atmosphere.alt2pres(altitude) / 101325
        features['elevation_adjusted_radiation'] = float(features['clear_sky_ghi'] * pressure_ratio)

        # Sun progress features
        day_length = (sunset - sunrise).total_seconds()
        time_since_sunrise = (target_time - sunrise).total_seconds()
        sun_progress = (time_since_sunrise / day_length) * 2 * np.pi

        features['sun_progress_sin'] = float(np.sin(sun_progress))
        features['sun_progress_cos'] = float(np.cos(sun_progress))

        features_list.append(features)

    # Create DataFrame
    feature_df = pd.DataFrame(features_list)

    # Ensure proper column order
    cols = ['idx'] + [col for col in feature_df.columns if col != 'idx']
    feature_df = feature_df[cols]

    return feature_df

generate_solar_features(train_windows[0])
## Time Features
 
import pandas as pd
import numpy as np
from datetime import datetime
import holidays

def encode_cyclical(value, max_val):
    """Encode a cyclical feature using sine and cosine"""
    sin = np.sin(2 * np.pi * value / max_val)
    cos = np.cos(2 * np.pi * value / max_val)
    return sin, cos

def get_season(month):
    """Get season ID (1: Spring, 2: Summer, 3: Autumn, 4: Winter)"""
    if month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    elif month in [9, 10, 11]:
        return 3
    else:  # [12, 1, 2]
        return 4

def get_time_of_day(hour):
    """Get time of day indicators"""
    is_morning = 1 if 5 <= hour < 12 else 0
    is_afternoon = 1 if 12 <= hour < 17 else 0
    is_evening = 1 if 17 <= hour < 20 else 0
    is_night = 1 if hour >= 20 or hour < 5 else 0
    return is_morning, is_afternoon, is_evening, is_night

def generate_time_features(window_data):
    """Generate all time-based features"""
    location_code = window_data['LocationCode'].iloc[0]
    prediction_date = window_data['DateTime'].iloc[-1].date()

    features_list = []

    # Generate prediction times
    prediction_times = pd.date_range(
        start=f"{prediction_date} 09:00:00",
        end=f"{prediction_date} 16:50:00",
        freq='10T'
    )

    # Create Taiwan holidays calendar
    tw_holidays = holidays.TW()

    for target_time in prediction_times:
        features = {}

        # Create ID
        features['idx'] = int(f"{target_time.strftime('%Y%m%d%H%M')}{location_code:02d}")

        # Hour features (0-23)
        hour_sin, hour_cos = encode_cyclical(target_time.hour + target_time.minute/60, 24)
        features['hour_sin'] = hour_sin
        features['hour_cos'] = hour_cos

        # Day of year features (1-366)
        day_of_year = target_time.dayofyear
        dayofyear_sin, dayofyear_cos = encode_cyclical(day_of_year, 366)
        features['dayofyear_sin'] = dayofyear_sin
        features['dayofyear_cos'] = dayofyear_cos

        # Month features (1-12)
        month_sin, month_cos = encode_cyclical(target_time.month, 12)
        features['month_sin'] = month_sin
        features['month_cos'] = month_cos
        features['month_id'] = target_time.month

        # Season features
        season_id = get_season(target_time.month)
        features['season_id'] = season_id
        features['is_spring'] = 1 if season_id == 1 else 0
        features['is_summer'] = 1 if season_id == 2 else 0
        features['is_autumn'] = 1 if season_id == 3 else 0
        features['is_winter'] = 1 if season_id == 4 else 0

        # Progress features
        # Day progress (0-1)
        minutes_since_midnight = target_time.hour * 60 + target_time.minute
        features['day_progress'] = minutes_since_midnight / (24 * 60)

        # Year progress (0-1)
        days_in_year = 366 if target_time.is_leap_year else 365
        features['year_progress'] = (day_of_year - 1) / days_in_year

        # Time of day indicators
        is_morning, is_afternoon, is_evening, is_night = get_time_of_day(target_time.hour)
        features['is_morning'] = is_morning
        features['is_afternoon'] = is_afternoon
        features['is_evening'] = is_evening
        features['is_night'] = is_night

        # Business hour and holiday features
        features['is_business_hour'] = 1 if 8 <= target_time.hour < 18 else 0
        features['is_holiday'] = 1 if target_time.date() in tw_holidays else 0
        features['is_weekend'] = 1 if target_time.weekday() >= 5 else 0

        features_list.append(features)

    # Create DataFrame
    feature_df = pd.DataFrame(features_list)

    # Ensure proper column order
    cols = ['idx'] + [col for col in feature_df.columns if col != 'idx']
    feature_df = feature_df[cols]

    return feature_df

generate_time_features(train_windows[0])
## Interaction Features
 
import numpy as np
import pandas as pd
from scipy.stats import circmean
import metpy.calc as mpcalc
from metpy.units import units

def calculate_weighted_circular_mean(angles, weights):
    """Calculate weighted circular mean of angles in degrees"""
    # Convert to radians
    angles_rad = np.radians(angles)

    # Calculate weighted components
    x = np.sum(weights * np.cos(angles_rad))
    y = np.sum(weights * np.sin(angles_rad))

    # Calculate mean angle
    mean_angle = np.arctan2(y, x)

    # Convert back to degrees
    return np.degrees(mean_angle) % 360

def calculate_wind_steadiness(directions, speeds):
    """Calculate wind direction steadiness"""
    if len(directions) < 2:
        return 0

    # Convert to radians for circular statistics
    rad_directions = np.radians(directions)

    # Calculate weighted mean direction
    mean_direction = np.radians(calculate_weighted_circular_mean(directions, speeds))

    # Calculate deviation from mean direction
    deviations = np.abs(np.angle(np.exp(1j * (rad_directions - mean_direction))))

    # Normalize and invert so higher values indicate more steadiness
    steadiness = 1 - np.mean(deviations) / np.pi
    return steadiness

def get_beaufort_category(speed):
    """Convert wind speed to Beaufort scale category"""
    beaufort_scale = [
        (0, 0.3), (1, 1.5), (2, 3.3), (3, 5.5),
        (4, 7.9), (5, 10.7), (6, 13.8), (7, 17.1),
        (8, 20.7), (9, 24.4), (10, 28.4), (11, 32.6), (12, float('inf'))
    ]

    for category, max_speed in beaufort_scale:
        if speed <= max_speed:
            return category
    return 12


def calculate_vapor_pressure(temperature, humidity):
    """Calculate vapor pressure deficit"""
    temperature = units.Quantity(temperature, 'degC')
    saturation_vapor_pressure = mpcalc.saturation_vapor_pressure(temperature)
    actual_vapor_pressure = saturation_vapor_pressure * humidity
    vapor_deficit = saturation_vapor_pressure - actual_vapor_pressure
    return vapor_deficit.magnitude

def calculate_heat_index(temperature, humidity):
    """
    Calculate heat index using simplified Rothfusz regression
    Temperature in Celsius, humidity in decimal form (0-1)
    """
    # Convert Celsius to Fahrenheit
    temperature_f = temperature * 9/5 + 32

    # Convert decimal humidity to percentage
    rh = humidity * 100

    # Simplified Rothfusz regression
    hi = 0.5 * (temperature_f + 61.0 + ((temperature_f - 68.0) * 1.2) + (rh * 0.094))

    # If temperature is high enough, use full regression
    if temperature_f >= 80:
        hi = -42.379 + 2.04901523 * temperature_f + 10.14333127 * rh \
             - 0.22475541 * temperature_f * rh - 0.00683783 * temperature_f * temperature_f \
             - 0.05481717 * rh * rh + 0.00122874 * temperature_f * temperature_f * rh \
             + 0.00085282 * temperature_f * rh * rh - 0.00000199 * temperature_f * temperature_f * rh * rh

    # Convert back to Celsius
    hi_celsius = (hi - 32) * 5/9

    return float(hi_celsius)

def normalize_interaction(x, y):
    """Calculate normalized interaction between two variables"""
    x_norm = (x - x.mean()) / (x.std() + 1e-8)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    # Return the mean of the interaction to get a scalar value
    return float(np.mean(x_norm * y_norm))

def get_wind_direction_category(degrees):
    """Convert wind direction degrees to 8 cardinal directions with error handling"""
    try:
        if pd.isna(degrees):
            return 'N'  # Default value

        degrees = float(degrees) % 360
        bins = np.arange(-22.5, 360, 45)
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
        bin_index = np.digitize(degrees, bins)
        return directions[min(bin_index, len(directions) - 1)]
    except:
        return 'N'  # Return default value if any error occurs

def get_safe_value(data_series, default=0.0):
    """Safely get value from series with error handling"""
    try:
        if len(data_series) > 0:
            value = data_series.iloc[-1]
            return float(value) if pd.notnull(value) else default
        return default
    except:
        return default

def generate_interaction_features(window_data, minute_weather):
    """Generate interaction and polynomial features with robust error handling"""
    location_code = window_data['LocationCode'].iloc[0]
    prediction_date = window_data['DateTime'].iloc[-1].date()

    features_list = []

    # Generate prediction times
    prediction_times = pd.date_range(
        start=f"{prediction_date} 09:00:00",
        end=f"{prediction_date} 16:50:00",
        freq='10T'
    )

    for target_time in prediction_times:
        features = {}
        features['idx'] = int(f"{target_time.strftime('%Y%m%d%H%M')}{location_code:02d}")

        try:
            # Get relevant weather data window
            weather_window = minute_weather[
                (minute_weather['datetime'] >= target_time - pd.Timedelta(hours=1)) &
                (minute_weather['datetime'] <= target_time)
                ]

            if len(weather_window) > 0:
                current_weather = weather_window.iloc[-1]

                # 1. Wind Characteristics
                features['wind_direction_binned'] = get_wind_direction_category(
                    get_safe_value(current_weather['wind_direction'])
                )

                wind_speed = get_safe_value(current_weather['wind_speed'])
                features['wind_speed_category'] = int(get_beaufort_category(wind_speed))

                # Gust wind ratio
                if wind_speed > 0:
                    gust_speed = get_safe_value(current_weather['max_gust_speed'])
                    features['gust_wind_ratio'] = float(gust_speed / wind_speed)
                else:
                    features['gust_wind_ratio'] = 0.0

                # 2. Wind Stability
                try:
                    features['direction_steadiness'] = float(calculate_wind_steadiness(
                        weather_window['wind_direction'].fillna(0).values,
                        weather_window['wind_speed'].fillna(0).values
                    ))
                except:
                    features['direction_steadiness'] = 0.0

                # Gust frequency
                try:
                    gust_threshold = weather_window['wind_speed'].mean() * 1.5
                    features['gust_frequency'] = float(np.mean(
                        weather_window['max_gust_speed'] > gust_threshold
                    ))
                except:
                    features['gust_frequency'] = 0.0

                # 3. Temperature-Humidity Features
                try:
                    temp = get_safe_value(current_weather['temperature'])
                    humid = get_safe_value(current_weather['humidity'])
                    features['vapor_saturation_deficit'] = float(calculate_vapor_pressure(
                        temp, humid
                    ))
                except:
                    features['vapor_saturation_deficit'] = 0.0

                try:
                    features['relative_humidity_slope'] = float(np.polyfit(
                        range(len(weather_window)),
                        weather_window['humidity'].fillna(method='ffill').fillna(0).values,
                        1
                    )[0])
                except:
                    features['relative_humidity_slope'] = 0.0

                # Heat stress and comfort metrics
                try:
                    features['heat_stress_index'] = calculate_heat_index(
                        temp, humid
                    )
                except:
                    features['heat_stress_index'] = 0.0

                # Feels like temperature
                try:
                    features['feels_like_temp'] = float(temp + (
                            0.348 * humid * 100 -
                            0.7 * wind_speed +
                            0.7 * humid *
                            (temp - 26)
                    ))
                except:
                    features['feels_like_temp'] = temp

                # 4. Pressure Characteristics
                try:
                    pressure_change = current_weather['pressure'] - weather_window['pressure'].iloc[0]
                    features['pressure_change_category'] = int(np.sign(pressure_change))
                    features['pressure_variation_intensity'] = float(np.std(
                        weather_window['pressure'].fillna(method='ffill').fillna(0)
                    ))
                except:
                    features['pressure_change_category'] = 0
                    features['pressure_variation_intensity'] = 0.0

                # 5. Normalized Interactions
                for interaction_pair in [
                    ('temperature', 'humidity', 'temp_humidity_norm'),
                    ('temperature', 'pressure', 'temp_pressure_norm'),
                    ('wind_speed', 'temperature', 'wind_temp_norm')
                ]:
                    try:
                        features[interaction_pair[2]] = normalize_interaction(
                            weather_window[interaction_pair[0]].fillna(method='ffill').fillna(0).values,
                            weather_window[interaction_pair[1]].fillna(method='ffill').fillna(0).values
                        )
                    except:
                        features[interaction_pair[2]] = 0.0

            else:
                # Default values if no weather data available
                features.update({
                    'wind_direction_binned': 'N',
                    'wind_speed_category': 0,
                    'gust_wind_ratio': 0.0,
                    'direction_steadiness': 0.0,
                    'gust_frequency': 0.0,
                    'vapor_saturation_deficit': 0.0,
                    'relative_humidity_slope': 0.0,
                    'heat_stress_index': 0.0,
                    'feels_like_temp': 0.0,
                    'pressure_change_category': 0,
                    'pressure_variation_intensity': 0.0,
                    'temp_humidity_norm': 0.0,
                    'temp_pressure_norm': 0.0,
                    'wind_temp_norm': 0.0
                })

        except Exception as e:
            # If any error occurs, use default values
            features.update({
                'wind_direction_binned': 'N',
                'wind_speed_category': 0,
                'gust_wind_ratio': 0.0,
                'direction_steadiness': 0.0,
                'gust_frequency': 0.0,
                'vapor_saturation_deficit': 0.0,
                'relative_humidity_slope': 0.0,
                'heat_stress_index': 0.0,
                'feels_like_temp': 0.0,
                'pressure_change_category': 0,
                'pressure_variation_intensity': 0.0,
                'temp_humidity_norm': 0.0,
                'temp_pressure_norm': 0.0,
                'wind_temp_norm': 0.0
            })

        features_list.append(features)

    # Create DataFrame
    feature_df = pd.DataFrame(features_list)

    # Ensure proper column order
    cols = ['idx'] + [col for col in feature_df.columns if col != 'idx']
    feature_df = feature_df[cols]

    return feature_df

generate_interaction_features(train_windows[0], weather_min)
## AQI Features
 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

def safe_convert_to_float(series):
    """Safely convert series to float, handling 'x' and other invalid values"""
    return pd.to_numeric(series.replace({'x': np.nan}), errors='coerce')

def calculate_slope_and_direction(series):
    """Calculate slope and trend direction from a time series"""
    if len(series.dropna()) < 2:
        return 0, 0

    try:
        # Get valid data points
        valid_data = series.dropna()
        x = np.arange(len(valid_data))

        # Calculate slope using linear regression
        slope, _, _, _, _ = stats.linregress(x, valid_data)

        # Determine trend direction (1: increasing, -1: decreasing, 0: stable)
        trend_direction = np.sign(slope)

        return slope, trend_direction
    except:
        return 0, 0

def calculate_window_stats(data, col, include_extended_stats=True):
    """Calculate statistical features for a window of data"""
    features = {}

    # Safely convert to float
    series = safe_convert_to_float(data[col])
    if len(series.dropna()) == 0:
        if include_extended_stats:
            return {'mean': 0, 'std': 0, 'peak': 0, 'slope': 0, 'trend_direction': 0}
        else:
            return {'mean': 0, 'peak': 0}

    # Basic stats
    features['mean'] = series.mean()
    features['peak'] = series.max()

    # Extended stats if requested
    if include_extended_stats:
        features['std'] = series.std() if len(series.dropna()) > 1 else 0
        slope, trend_direction = calculate_slope_and_direction(series)
        features['slope'] = slope
        features['trend_direction'] = trend_direction

    return features

def calculate_day_over_day_change(current_stats, previous_stats):
    """Calculate day-over-day changes for relevant metrics"""
    changes = {}
    for metric in ['mean', 'peak']:
        if previous_stats[metric] != 0:
            changes[f'{metric}_change'] = (current_stats[metric] - previous_stats[metric]) / previous_stats[metric]
        else:
            changes[f'{metric}_change'] = 0
    return changes

def generate_aqi_features(window_data, aqi_data):
    """Generate optimized AQI features for the prediction window"""
    # Get location and date information
    location_code = window_data['LocationCode'].iloc[0]
    prediction_date = window_data['DateTime'].iloc[-1].date()

    # Define pollutant groups
    main_pollutants = ['PM2.5', 'PM10', 'NOx', 'NO2']
    secondary_pollutants = ['SO2', 'CO']

    # Generate features for each prediction time point
    all_features = []
    prediction_times = pd.date_range(
        start=f"{prediction_date} 09:00:00",
        end=f"{prediction_date} 16:50:00",
        freq='10T'
    )

    for target_time in prediction_times:
        features = {}
        features['idx'] = int(f"{target_time.strftime('%Y%m%d%H%M')}{location_code:02d}")

        # 1. Current Data
        current_hour = target_time.floor('H')
        current_data = aqi_data[aqi_data['monitordate'] == current_hour]

        if len(current_data) > 0:
            # Raw values and normalization
            for col in main_pollutants + secondary_pollutants:
                value = safe_convert_to_float(current_data[col]).iloc[0]
                features[f'aqi_{col}_current'] = value

                # Normalize using min-max scaling from the last 24 hours
                last_24h = aqi_data[
                    (aqi_data['monitordate'] >= current_hour - timedelta(hours=24)) &
                    (aqi_data['monitordate'] <= current_hour)
                    ]
                col_values = safe_convert_to_float(last_24h[col])
                col_min, col_max = col_values.min(), col_values.max()
                if col_max > col_min:
                    features[f'aqi_{col}_normalized'] = (value - col_min) / (col_max - col_min)
                else:
                    features[f'aqi_{col}_normalized'] = 0

            # Ratios
            features['aqi_PM25_PM10_ratio'] = (
                features['aqi_PM2.5_current'] / features['aqi_PM10_current']
                if features['aqi_PM10_current'] != 0 else 0
            )
            features['aqi_NOx_NO2_ratio'] = (
                features['aqi_NOx_current'] / features['aqi_NO2_current']
                if features['aqi_NO2_current'] != 0 else 0
            )

        # 2. Rolling Windows (3h, 6h)
        for hours in [3, 6]:
            window_start = target_time - timedelta(hours=hours)
            window_data = aqi_data[
                (aqi_data['monitordate'] >= window_start) &
                (aqi_data['monitordate'] <= target_time)
                ]

            # Main pollutants with extended stats
            for col in main_pollutants:
                stats = calculate_window_stats(window_data, col, include_extended_stats=True)
                features.update({f'aqi_{col}_{hours}h_{k}': v for k, v in stats.items()
                                 if k in ['mean', 'std', 'peak', 'slope']})

            # Secondary pollutants with basic stats
            for col in secondary_pollutants:
                stats = calculate_window_stats(window_data, col, include_extended_stats=False)
                features.update({f'aqi_{col}_{hours}h_{k}': v for k, v in stats.items()})

        # 3. Today's Morning (7:00-9:00)
        morning_start = target_time.replace(hour=7, minute=0)
        morning_end = target_time.replace(hour=8, minute=59)
        morning_data = aqi_data[
            (aqi_data['monitordate'] >= morning_start) &
            (aqi_data['monitordate'] <= morning_end)
            ]

        # Calculate morning stats
        for col in main_pollutants:
            stats = calculate_window_stats(morning_data, col, include_extended_stats=True)
            features.update({f'aqi_{col}_morning_{k}': v for k, v in stats.items()})

        for col in secondary_pollutants:
            stats = calculate_window_stats(morning_data, col, include_extended_stats=False)
            features.update({f'aqi_{col}_morning_{k}': v for k, v in stats.items()})

        # 4. Yesterday's Morning and Full Day
        yesterday = target_time.date() - timedelta(days=1)

        # Yesterday morning
        yesterday_morning_start = datetime.combine(yesterday, datetime.min.time().replace(hour=7))
        yesterday_morning_end = datetime.combine(yesterday, datetime.min.time().replace(hour=8, minute=59))
        yesterday_morning_data = aqi_data[
            (aqi_data['monitordate'] >= yesterday_morning_start) &
            (aqi_data['monitordate'] <= yesterday_morning_end)
            ]

        # Yesterday full day
        yesterday_start = datetime.combine(yesterday, datetime.min.time())
        yesterday_end = datetime.combine(yesterday, datetime.max.time())
        yesterday_data = aqi_data[
            (aqi_data['monitordate'] >= yesterday_start) &
            (aqi_data['monitordate'] <= yesterday_end)
            ]

        # Calculate yesterday's stats and day-over-day changes
        for col in main_pollutants:
            # Morning stats
            yesterday_morning_stats = calculate_window_stats(yesterday_morning_data, col, include_extended_stats=True)
            features.update({f'aqi_{col}_yesterday_morning_{k}': v for k, v in yesterday_morning_stats.items()})

            # Full day stats
            yesterday_stats = calculate_window_stats(yesterday_data, col, include_extended_stats=True)
            features.update({f'aqi_{col}_yesterday_{k}': v for k, v in yesterday_stats.items()})

            # Day-over-day changes
            current_morning_stats = calculate_window_stats(morning_data, col, include_extended_stats=True)
            morning_changes = calculate_day_over_day_change(current_morning_stats, yesterday_morning_stats)
            features.update({f'aqi_{col}_morning_{k}': v for k, v in morning_changes.items()})

        all_features.append(features)

    # Create DataFrame with all features
    feature_df = pd.DataFrame(all_features)

    # Ensure idx is the first column
    cols = ['idx'] + [col for col in feature_df.columns if col != 'idx']
    feature_df = feature_df[cols]

    return feature_df

# Load AQI data
aqi_data = pd.read_csv('AQI.csv')
aqi_data['monitordate'] = pd.to_datetime(aqi_data['monitordate'])

# Generate features for a window
generate_aqi_features(train_windows[0], aqi_data)
## Advanced Weather and Performance Features
 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import metpy.calc as mpcalc
from metpy.units import units

def calculate_dew_point(temperature, relative_humidity):
    """Calculate dew point temperature"""
    try:
        temperature = units.Quantity(temperature, 'degC')
        rh = relative_humidity * 100
        return mpcalc.dewpoint_from_relative_humidity(temperature, rh).magnitude
    except:
        return np.nan

def estimate_cloud_cover(temp_max, temp_min, humidity):
    """Estimate cloud cover using diurnal temperature range and humidity"""
    try:
        temp_range = temp_max - temp_min
        temp_range_factor = 1 - (temp_range / 15)  # Assume max clear sky temp range is 15°C
        humidity_factor = humidity
        cloud_cover = (temp_range_factor + humidity_factor) / 2
        return min(max(cloud_cover, 0), 1)
    except:
        return 0.5

def calculate_panel_temperature(ambient_temp, solar_radiation, wind_speed):
    """Calculate panel temperature using Faiman model"""
    try:
        # Faiman model coefficients
        u0, u1 = 25.0, 6.84  # Standard coefficients

        # Convert solar radiation from lux to W/m2 (approximate)
        solar_radiation_wm2 = solar_radiation * 0.0079  # Rough conversion

        # Calculate panel temperature
        delta_t = solar_radiation_wm2 / (u0 + u1 * wind_speed)
        panel_temp = ambient_temp + delta_t

        return panel_temp
    except:
        return ambient_temp

def calculate_panel_efficiency(panel_temp, base_efficiency=0.15):
    """Calculate panel efficiency based on temperature"""
    try:
        temp_coefficient = -0.0045  # Typical temperature coefficient (%/°C)
        temp_diff = panel_temp - 25  # Difference from STC temperature (25°C)
        efficiency = base_efficiency * (1 + temp_coefficient * temp_diff)
        return max(efficiency, 0)
    except:
        return base_efficiency

def estimate_soiling_loss(pm25, humidity, days_since_rain=1):
    """Estimate soiling loss based on air quality and weather"""
    try:
        # Base soiling rate based on PM2.5
        base_soiling = min(pm25 / 500, 0.05)  # Max 5% loss at PM2.5 = 500

        # Humidity effect (higher humidity = more particle adhesion)
        humidity_factor = humidity ** 0.5

        # Days since rain effect
        time_factor = min(days_since_rain / 14, 1)  # Saturate at 14 days

        total_loss = base_soiling * humidity_factor * time_factor
        return min(total_loss, 0.15)  # Cap at 15% loss
    except:
        return 0.01

def generate_advanced_features(window_data, weather_min, aqi_data):
    """Generate advanced weather and performance features with robust error handling"""
    location_code = window_data['LocationCode'].iloc[0]
    prediction_date = window_data['DateTime'].iloc[-1].date()

    features_list = []
    prediction_times = pd.date_range(
        start=f"{prediction_date} 09:00:00",
        end=f"{prediction_date} 16:50:00",
        freq='10T'
    )

    # Pre-calculate morning and yesterday data
    morning_data = window_data[
        (window_data['DateTime'].dt.date == prediction_date) &
        (window_data['DateTime'].dt.hour.between(7, 8))
        ]

    yesterday = prediction_date - timedelta(days=1)
    yesterday_data = window_data[
        window_data['DateTime'].dt.date == yesterday
        ]

    morning_temp_mean = morning_data['Temperature(°C)'].mean()
    morning_power_mean = morning_data['Power(mW)'].mean()

    for target_time in prediction_times:
        features = {}
        features['idx'] = int(f"{target_time.strftime('%Y%m%d%H%M')}{location_code:02d}")

        try:
            # Get nearest data point instead of exact window
            nearest_time_idx = window_data['DateTime'].sub(target_time).abs().idxmin()
            current_data = window_data.loc[nearest_time_idx:nearest_time_idx+2]

            if len(current_data) > 0:
                # Current measurements with error handling
                temp = float(current_data['Temperature(°C)'].iloc[0])
                humidity = float(current_data['Humidity(%)'].iloc[0]) / 100
                wind_speed = float(current_data['WindSpeed(m/s)'].iloc[0])
                sunlight = float(current_data['Sunlight(Lux)'].iloc[0])
                power = float(current_data['Power(mW)'].iloc[0])

                # 1. Meteorological Features
                features['dew_point'] = calculate_dew_point(temp, humidity)

                # Wet bulb temperature (simplified)
                features['wet_bulb_temp'] = temp * np.arctan(0.151977 * (humidity * 100 + 8.313659)**0.5) + \
                                            np.arctan(temp + humidity * 100) - \
                                            np.arctan(humidity * 100 - 1.676331) + \
                                            0.00391838 * (humidity * 100)**1.5 * \
                                            np.arctan(0.023101 * humidity * 100) - 4.686035

                # Wind chill (for temperatures below 10°C)
                features['wind_chill'] = 13.12 + 0.6215 * temp - 11.37 * (wind_speed * 3.6)**0.16 + \
                                         0.3965 * temp * (wind_speed * 3.6)**0.16 if temp < 10 else temp

                # Get temperature range for cloud cover
                temp_range = window_data[
                    (window_data['DateTime'].dt.date == prediction_date) &
                    (window_data['DateTime'].dt.hour < target_time.hour)
                    ]['Temperature(°C)']

                if len(temp_range) > 0:
                    temp_max = temp_range.max()
                    temp_min = temp_range.min()
                    features['estimated_cloud_cover'] = estimate_cloud_cover(temp_max, temp_min, humidity)
                else:
                    features['estimated_cloud_cover'] = 0.5

                # 2. Panel Performance Features
                panel_temp = calculate_panel_temperature(temp, sunlight, wind_speed)
                features['panel_temperature'] = panel_temp
                features['panel_efficiency_factor'] = calculate_panel_efficiency(panel_temp)

                # Thermal loss coefficient with safety check
                if sunlight > 0:
                    features['thermal_loss_coef'] = (panel_temp - temp) / sunlight
                else:
                    features['thermal_loss_coef'] = 0

                # Get AQI data
                current_hour = target_time.floor('H')
                current_aqi = aqi_data[aqi_data['monitordate'] == current_hour]

                if len(current_aqi) > 0:
                    pm25 = float(pd.to_numeric(current_aqi['PM2.5'].iloc[0], errors='coerce'))

                    # Check for recent rain
                    recent_weather = weather_min[
                        (weather_min['datetime'] >= target_time - timedelta(hours=24)) &
                        (weather_min['datetime'] <= target_time)
                        ]
                    days_since_rain = 0 if len(recent_weather) > 0 and \
                                           (recent_weather['precipitation'] > 0).any() else 1

                    features['estimated_soiling_loss'] = estimate_soiling_loss(
                        pm25, humidity, days_since_rain
                    )
                else:
                    features['estimated_soiling_loss'] = 0.01

                # 3. Cross-Window Performance Ratios
                # Morning ratios
                if pd.notnull(morning_temp_mean) and morning_temp_mean != 0:
                    features['morning_current_temp_ratio'] = temp / morning_temp_mean
                else:
                    features['morning_current_temp_ratio'] = 1

                if pd.notnull(morning_power_mean) and morning_power_mean != 0:
                    features['morning_current_power_ratio'] = power / morning_power_mean
                else:
                    features['morning_current_power_ratio'] = 1

                # Yesterday comparison
                yesterday_same_time = yesterday_data[
                    (yesterday_data['DateTime'].dt.hour == target_time.hour) &
                    (yesterday_data['DateTime'].dt.minute == target_time.minute)
                    ]

                if len(yesterday_same_time) > 0:
                    features['yesterday_current_temp_diff'] = \
                        temp - yesterday_same_time['Temperature(°C)'].iloc[0]
                    if yesterday_same_time['Power(mW)'].iloc[0] > 0:
                        features['yesterday_current_power_ratio'] = \
                            power / yesterday_same_time['Power(mW)'].iloc[0]
                    else:
                        features['yesterday_current_power_ratio'] = 1
                else:
                    features['yesterday_current_temp_diff'] = 0
                    features['yesterday_current_power_ratio'] = 1

            else:
                # Set default values if no current data available
                features.update({
                    'dew_point': 0,
                    'wet_bulb_temp': 0,
                    'wind_chill': 0,
                    'estimated_cloud_cover': 0.5,
                    'panel_temperature': 0,
                    'panel_efficiency_factor': 0.15,
                    'thermal_loss_coef': 0,
                    'estimated_soiling_loss': 0.01,
                    'morning_current_temp_ratio': 1,
                    'morning_current_power_ratio': 1,
                    'yesterday_current_temp_diff': 0,
                    'yesterday_current_power_ratio': 1
                })

        except Exception as e:
            # Set default values if any error occurs
            features.update({
                'dew_point': 0,
                'wet_bulb_temp': 0,
                'wind_chill': 0,
                'estimated_cloud_cover': 0.5,
                'panel_temperature': 0,
                'panel_efficiency_factor': 0.15,
                'thermal_loss_coef': 0,
                'estimated_soiling_loss': 0.01,
                'morning_current_temp_ratio': 1,
                'morning_current_power_ratio': 1,
                'yesterday_current_temp_diff': 0,
                'yesterday_current_power_ratio': 1
            })

        features_list.append(features)

    # Create DataFrame with all features
    feature_df = pd.DataFrame(features_list)

    # Ensure idx is the first column
    cols = ['idx'] + [col for col in feature_df.columns if col != 'idx']
    feature_df = feature_df[cols]

    return feature_df

# Generate advanced features
generate_advanced_features(train_windows[0], weather_min, aqi_data)
 
raise Exception("Stop here")
# Combine All Feature
 
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def process_window(window, weather_hour, weather_min):
    """Process a single window to generate all features"""
    # Generate all types of features
    local_climate = generate_local_climate_features(window, weather_hour, weather_min)
    historical = generate_historical_features(window)
    solar = generate_solar_features(window)
    time = generate_time_features(window)
    interaction = generate_interaction_features(window, weather_min)
    aqi = generate_aqi_features(window, aqi_data)
    advanced = generate_advanced_features(window, weather_min, aqi_data)
    # multi_station = generate_multi_station_features(window, all_station_data)

    # Merge all features based on idx
    features = local_climate.merge(historical, on='idx', how='left')
    features = features.merge(solar, on='idx', how='left')
    features = features.merge(time, on='idx', how='left')
    features = features.merge(interaction, on='idx', how='left')
    features = features.merge(aqi, on='idx', how='left')
    features = features.merge(advanced, on='idx', how='left')
    # features = features.merge(multi_station, on='idx', how='left')

    # generate date and location features
    features['date'] = window['DateTime'].iloc[0].date()
    features['location'] = window['LocationCode'].iloc[0]

    return features

def create_final_datasets(train_windows, test_windows, train_labels, weather_hour, weather_min):
    """Create final training and testing datasets"""
    print("Processing training windows...")
    train_features_list = []
    for window in tqdm(train_windows, desc="Training"):
        try:
            features = process_window(window, weather_hour, weather_min)
            train_features_list.append(features)
        except Exception as e:
            print(f"Error processing training window: {e}")
            continue

    print("\nProcessing testing windows...")
    test_features_list = []
    for window in tqdm(test_windows, desc="Testing"):
        try:
            features = process_window(window, weather_hour, weather_min)
            test_features_list.append(features)
        except Exception as e:
            print(f"Error processing testing window: {e}")
            continue

    # Combine all features
    print("\nCombining features...")
    train_df = pd.concat(train_features_list, ignore_index=True)
    test_df = pd.concat(test_features_list, ignore_index=True)

    # Merge training labels
    print("Merging training labels...")
    train_df = train_df.merge(
        train_labels[['idx', 'Power(mW)']],
        on='idx',
        how='left'
    )

    # Sort by idx
    train_df = train_df.sort_values('idx').reset_index(drop=True)
    test_df = test_df.sort_values('idx').reset_index(drop=True)

    # Save to CSV
    print("Saving datasets...")
    train_df.to_csv('FE/train_1123.csv', index=False)
    test_df.to_csv('FE/test_1123.csv', index=False)

    print("\nDataset Statistics:")
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    print(f"Number of features: {len(train_df.columns) - 2}")  # -2 for idx and Power

    # Check for missing values
    train_nulls = train_df.isnull().sum()
    test_nulls = test_df.isnull().sum()

    if train_nulls.any() or test_nulls.any():
        print("\nWarning: Missing values detected!")
        print("\nTraining set missing values:")
        print(train_nulls[train_nulls > 0])
        print("\nTesting set missing values:")
        print(test_nulls[test_nulls > 0])

    return train_df, test_df

# Execute the pipeline
if __name__ == "__main__":
    print("Loading weather data...")
    # Load weather data (only once)
    weather_hour = pd.read_csv('hualien_weather_hour.csv')
    weather_hour['obsTime'] = pd.to_datetime(weather_hour['obsTime'])
    weather_hour = weather_hour[weather_hour['station_id'] == 'C0Z100']

    weather_min = pd.read_csv('hualien_weather_min.csv')
    weather_min['datetime'] = pd.to_datetime(weather_min['datetime'])

    print("Starting feature generation pipeline...")
    train_df, test_df = create_final_datasets(
        train_windows,
        test_windows,
        train_labels,
        weather_hour,
        weather_min
    )

    print("\nPipeline completed successfully!")
## Extention: Add himawari data
 
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

def convert_to_feature_idx(row):
    """Convert datetime and location_code to feature IDX format"""
    dt = pd.to_datetime(row['datetime'])
    location = int(row['location_code'])
    return int(f"{dt.strftime('%Y%m%d%H%M')}{location:02d}")

def generate_himawari_features(feature_df, himawari_data):
    """
    Generate Himawari satellite features and merge with existing features
    
    Args:
        feature_df: DataFrame containing existing features (train or test)
        himawari_data: DataFrame containing Himawari satellite data
        
    Returns:
        DataFrame with added Himawari features
    """
    print("Processing Himawari satellite data...")

    # Create copy of feature_df to avoid modifying original
    result_df = feature_df.copy()

    # Convert datetime in himawari_data to idx format for merging
    himawari_data['idx'] = himawari_data.apply(convert_to_feature_idx, axis=1)

    # Rename columns to more descriptive names
    feature_columns = {
        # Band 03 (Visible) features
        'B03_mean': 'himawari_visible_mean',
        'B03_std': 'himawari_visible_std',
        'B03_min': 'himawari_visible_min',
        'B03_max': 'himawari_visible_max',
        'B03_cloud_coverage': 'himawari_visible_cloud_coverage',
        'B03_clarity_index': 'himawari_visible_clarity_index',
        'B03_brightness_contrast': 'himawari_visible_brightness_contrast',

        # Band 13 (IR) features
        'B13_mean': 'himawari_ir13_mean',
        'B13_std': 'himawari_ir13_std',
        'B13_min': 'himawari_ir13_min',
        'B13_max': 'himawari_ir13_max',
        'B13_cloud_temp_mean': 'himawari_ir13_cloud_temp_mean',
        'B13_clear_sky_temp': 'himawari_ir13_clear_sky_temp',
        'B13_cloud_coverage': 'himawari_ir13_cloud_coverage',

        # Band 14 (IR) features
        'B14_mean': 'himawari_ir14_mean',
        'B14_std': 'himawari_ir14_std',
        'B14_min': 'himawari_ir14_min',
        'B14_max': 'himawari_ir14_max',
        'B14_cloud_temp_mean': 'himawari_ir14_cloud_temp_mean',
        'B14_clear_sky_temp': 'himawari_ir14_clear_sky_temp',
        'B14_cloud_coverage': 'himawari_ir14_cloud_coverage'
    }

    # Rename columns
    himawari_features = himawari_data.rename(columns=feature_columns)

    # Calculate additional derived features
    himawari_features['himawari_ir_temp_diff'] = (
            himawari_features['himawari_ir14_mean'] - himawari_features['himawari_ir13_mean']
    )

    himawari_features['himawari_total_cloud_coverage'] = (
        himawari_features[['himawari_visible_cloud_coverage',
                           'himawari_ir13_cloud_coverage',
                           'himawari_ir14_cloud_coverage']].mean(axis=1)
    )

    # Calculate cloud type features
    himawari_features['himawari_high_cloud_indicator'] = np.where(
        (himawari_features['himawari_ir13_mean'] < 240) &
        (himawari_features['himawari_ir14_mean'] < 240),
        1, 0
    )

    himawari_features['himawari_middle_cloud_indicator'] = np.where(
        (himawari_features['himawari_ir13_mean'].between(240, 260)) &
        (himawari_features['himawari_ir14_mean'].between(240, 260)),
        1, 0
    )

    himawari_features['himawari_low_cloud_indicator'] = np.where(
        (himawari_features['himawari_ir13_mean'] > 260) &
        (himawari_features['himawari_ir14_mean'] > 260),
        1, 0
    )

    # Calculate cloud development features
    himawari_features['himawari_cloud_development_index'] = (
            himawari_features['himawari_ir14_max'] - himawari_features['himawari_ir14_min']
    )

    # Calculate solar interference indicators
    himawari_features['himawari_solar_interference'] = (
            himawari_features['himawari_visible_brightness_contrast'] *
            himawari_features['himawari_visible_clarity_index']
    )

    # Select columns for merging
    merge_columns = ['idx'] + [col for col in himawari_features.columns
                               if col.startswith('himawari_')]

    # Merge with existing features
    print("Merging Himawari features with existing features...")
    result_df = result_df.merge(
        himawari_features[merge_columns],
        on='idx',
        how='left'
    )

    # Fill missing values with appropriate defaults
    print("Handling missing values...")
    for col in result_df.columns:
        if col.startswith('himawari_'):
            if 'indicator' in col or 'cloud_coverage' in col:
                result_df[col].fillna(0, inplace=True)
            elif 'temp' in col:
                result_df[col].fillna(result_df[col].mean(), inplace=True)
            else:
                result_df[col].fillna(0, inplace=True)

    print("Himawari feature generation completed!")
    return result_df

# Usage example:
def process_datasets():
    """Process both train and test datasets with Himawari features"""
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv('FE/train_1121.csv')
    test_df = pd.read_csv('FE/test_1121.csv')
    himawari_data = pd.read_csv('himawari_data.csv')

    # Process train and test sets
    print("\nProcessing training set...")
    train_df_with_himawari = generate_himawari_features(train_df, himawari_data)

    print("\nProcessing test set...")
    test_df_with_himawari = generate_himawari_features(test_df, himawari_data)

    # Save results
    print("\nSaving results...")
    train_df_with_himawari.to_csv('train_level_1_.csv', index=False)
    test_df_with_himawari.to_csv('train_level_1_.csv', index=False)

    print("\nFeature statistics:")
    print(f"Training set shape: {train_df_with_himawari.shape}")
    print(f"Testing set shape: {test_df_with_himawari.shape}")

    # Check for any missing values
    train_nulls = train_df_with_himawari.isnull().sum()
    test_nulls = test_df_with_himawari.isnull().sum()

    if train_nulls.any() or test_nulls.any():
        print("\nWarning: Missing values detected!")
        print("\nTraining set missing values:")
        print(train_nulls[train_nulls > 0])
        print("\nTest set missing values:")
        print(test_nulls[test_nulls > 0])

if __name__ == "__main__":
    process_datasets()
 
def generate_ground_temp_features(window_data, ground_temp_data):
    """Generate features from ground temperature data"""
    # Extract date and location from idx
    idx_str = str(window_data['idx'].iloc[0])
    location_code = int(idx_str[-2:])
    prediction_date = datetime.strptime(idx_str[:8], '%Y%m%d').date()

    features_list = []

    # Generate prediction times
    prediction_times = pd.date_range(
        start=f"{prediction_date} 09:00:00",
        end=f"{prediction_date} 16:50:00",
        freq='10T'
    )

    # Clean and prepare ground temp data
    try:
        # Handle potential encoding issues and data inconsistencies
        if '測站代碼' in ground_temp_data.columns:
            ground_temp_data = ground_temp_data.rename(columns={
                '測站代碼': 'station_id',
                '觀測時間': 'observation_time',
                '全天空日射量(MJ/m2)': 'total_radiation',
                '蒸發量(mm)': 'evaporation',
                '地溫0cm(℃)': 'ground_temp_0cm',
                '地溫5cm(℃)': 'ground_temp_5cm',
                '地溫10cm(℃)': 'ground_temp_10cm',
                '地溫20cm(℃)': 'ground_temp_20cm',
                '地溫50cm(℃)': 'ground_temp_50cm',
                '地溫100cm(℃)': 'ground_temp_100cm'
            })

        # Convert observation time to datetime
        ground_temp_data['observation_time'] = pd.to_datetime(ground_temp_data['observation_time'])

        # Convert numeric columns and handle missing values
        numeric_cols = ['total_radiation', 'evaporation'] + [f'ground_temp_{d}cm' for d in [0,5,10,20,50,100]]
        for col in numeric_cols:
            if col in ground_temp_data.columns:
                ground_temp_data[col] = pd.to_numeric(ground_temp_data[col], errors='coerce')

        depth_cols = [f'ground_temp_{d}cm' for d in [0,5,10,20,50,100]]
        radiation_cols = ['total_radiation', 'evaporation']

        for target_time in prediction_times:
            features = {}
            features['idx'] = int(f"{target_time.strftime('%Y%m%d%H%M')}{location_code:02d}")

            # Get current hour data
            current_hour = target_time.replace(minute=0)
            current_data = ground_temp_data[
                ground_temp_data['observation_time'] == current_hour
                ]

            # 1. Current Ground Temperature Features
            if len(current_data) > 0:
                # Basic temperature readings
                for col in depth_cols:
                    if col in current_data.columns:
                        features[f'{col}_current'] = float(current_data[col].iloc[0])

                # Temperature gradients
                depths = [0, 5, 10, 20, 50, 100]
                for i in range(len(depths)-1):
                    col1 = f'ground_temp_{depths[i]}cm'
                    col2 = f'ground_temp_{depths[i+1]}cm'
                    if col1 in current_data.columns and col2 in current_data.columns:
                        gradient = (current_data[col2].iloc[0] -
                                    current_data[col1].iloc[0]) / (depths[i+1] - depths[i])
                        features[f'temp_gradient_{depths[i]}_{depths[i+1]}'] = gradient

                # Radiation features
                for col in radiation_cols:
                    if col in current_data.columns:
                        features[f'{col}_current'] = float(current_data[col].fillna(0).iloc[0])

            # 2. Moving Windows Statistics (3h, 6h)
            for hours in [3, 6]:
                window_start = target_time - timedelta(hours=hours)
                window_data = ground_temp_data[
                    (ground_temp_data['observation_time'] >= window_start) &
                    (ground_temp_data['observation_time'] <= target_time)
                    ]

                if len(window_data) > 0:
                    # Ground temperature statistics
                    for col in depth_cols:
                        if col in window_data.columns:
                            prefix = f'{col}_{hours}h'
                            features[f'{prefix}_mean'] = float(window_data[col].mean())
                            features[f'{prefix}_std'] = float(window_data[col].std())
                            features[f'{prefix}_min'] = float(window_data[col].min())
                            features[f'{prefix}_max'] = float(window_data[col].max())

                            # Temperature change rate
                            temp_changes = window_data[col].diff()
                            features[f'{prefix}_change_rate'] = float(temp_changes.mean())

                    # Radiation statistics
                    for col in radiation_cols:
                        if col in window_data.columns:
                            prefix = f'{col}_{hours}h'
                            features[f'{prefix}_sum'] = float(window_data[col].fillna(0).sum())
                            features[f'{prefix}_mean'] = float(window_data[col].fillna(0).mean())

            # 3. Daily features
            today_data = ground_temp_data[
                ground_temp_data['observation_time'].dt.date == target_time.date()
                ]

            if len(today_data) > 0:
                for col in depth_cols:
                    if col in today_data.columns:
                        prefix = f'{col}_daily'
                        features[f'{prefix}_range'] = float(
                            today_data[col].max() - today_data[col].min()
                        )
                        try:
                            features[f'{prefix}_trend'] = float(np.polyfit(
                                range(len(today_data)),
                                today_data[col].values,
                                1
                            )[0])
                        except:
                            features[f'{prefix}_trend'] = 0.0

            features_list.append(features)

    except Exception as e:
        print(f"Error processing ground temperature data: {e}")
        return pd.DataFrame({'idx': []})  # Return empty DataFrame on error

    # Create DataFrame and handle missing values
    feature_df = pd.DataFrame(features_list)
    feature_df = feature_df.fillna(0)  # Fill any remaining NaN values

    # Ensure proper column order
    cols = ['idx'] + [col for col in feature_df.columns if col != 'idx']
    feature_df = feature_df[cols]

    return feature_df

def process_datasets_with_ground_temp():
    """Process both train and test datasets with ground temperature features"""
    print("Loading datasets...")
    train_df = pd.read_csv('train_level_1_.csv')
    test_df = pd.read_csv('test_train_level_1_.csv')

    # Read ground temperature data with explicit encoding and separator
    ground_temp_data = pd.read_csv('gt.csv', encoding='utf-8', sep=',', on_bad_lines='skip')
    print(f"Ground temperature data shape: {ground_temp_data.shape}")
    print("Ground temperature columns:", ground_temp_data.columns.tolist())

    print("\nProcessing training set...")
    unique_groups = train_df.groupby('idx').first().reset_index()
    train_features = []

    for _, group in tqdm(unique_groups.iterrows(), total=len(unique_groups)):
        window_features = generate_ground_temp_features(
            pd.DataFrame([group]),
            ground_temp_data
        )
        if len(window_features) > 0:
            train_features.append(window_features)

    if train_features:
        train_df_with_ground_temp = pd.concat(train_features, ignore_index=True)
        train_df_final = train_df.merge(train_df_with_ground_temp, on='idx', how='left')
    else:
        print("Warning: No ground temperature features generated for training set")
        train_df_final = train_df

    print("\nProcessing test set...")
    unique_groups = test_df.groupby('idx').first().reset_index()
    test_features = []

    for _, group in tqdm(unique_groups.iterrows(), total=len(unique_groups)):
        window_features = generate_ground_temp_features(
            pd.DataFrame([group]),
            ground_temp_data
        )
        if len(window_features) > 0:
            test_features.append(window_features)

    if test_features:
        test_df_with_ground_temp = pd.concat(test_features, ignore_index=True)
        test_df_final = test_df.merge(test_df_with_ground_temp, on='idx', how='left')
    else:
        print("Warning: No ground temperature features generated for test set")
        test_df_final = test_df

    print("\nSaving results...")
    train_df_final.to_csv('train_level_1.csv', index=False)
    test_df_final.to_csv('test_level_1.csv', index=False)

    print("\nFeature statistics:")
    print(f"Training set shape: {train_df_final.shape}")
    print(f"Testing set shape: {test_df_final.shape}")

    # Check for missing values
    train_nulls = train_df_final.isnull().sum()
    test_nulls = test_df_final.isnull().sum()

    if train_nulls.any() or test_nulls.any():
        print("\nWarning: Missing values detected!")
        print("\nTraining set missing values:")
        print(train_nulls[train_nulls > 0])
        print("\nTest set missing values:")
        print(test_nulls[test_nulls > 0])

if __name__ == "__main__":
    process_datasets_with_ground_temp()