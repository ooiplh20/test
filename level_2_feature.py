import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_level2_data(cleaned_data_path: str, upload_template_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare Level 2 training and test data
    """
    def load_station_data(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        return df

    def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
        # Basic time components
        df['l2_month'] = df['DateTime'].dt.month
        df['l2_day'] = df['DateTime'].dt.day
        df['l2_hour'] = df['DateTime'].dt.hour
        df['l2_minute'] = df['DateTime'].dt.minute

        # Cyclical encoding
        seconds_in_day = 24 * 60 * 60
        day_seconds = df['DateTime'].dt.hour * 3600 + df['DateTime'].dt.minute * 60
        df['l2_time_cos'] = np.cos(2 * np.pi * day_seconds / seconds_in_day)
        df['l2_time_sin'] = np.sin(2 * np.pi * day_seconds / seconds_in_day)

        # Solar time features
        df['l2_solar_hour'] = (df['DateTime'].dt.hour + df['DateTime'].dt.minute/60) - 6

        return df

    def create_station_pairs(station_data: Dict[int, pd.DataFrame], is_training: bool = True) -> pd.DataFrame:
        pairs = []

        # Get unique timestamps
        all_times = set()
        for df in station_data.values():
            all_times.update(df['DateTime'].unique())
        all_times = sorted(list(all_times))

        for timestamp in tqdm(all_times, desc='Processing timestamps'):
            for target_loc in station_data.keys():
                target_df = station_data[target_loc]
                target_time = target_df[target_df['DateTime'] == timestamp].copy()

                if target_time.empty:
                    continue

                for source_loc in station_data.keys():
                    if source_loc == target_loc:
                        continue

                    source_df = station_data[source_loc]
                    source_time = source_df[source_df['DateTime'] == timestamp].copy()

                    if source_time.empty:
                        continue

                    pair_data = {
                        'DateTime': timestamp,
                        'target_location': target_loc,
                        'source_location': source_loc,
                        'l2_wind': source_time['WindSpeed(m/s)'].iloc[0],
                        'l2_pressure': source_time['Pressure(hpa)'].iloc[0],
                        'l2_temperature': source_time['Temperature(°C)'].iloc[0],
                        'l2_humidity': source_time['Humidity(%)'].iloc[0],
                        'l2_sunlight': source_time['Sunlight(Lux)'].iloc[0],
                        'l2_source_power': source_time['Power(mW)'].iloc[0],
                    }

                    if is_training:
                        pair_data['target_power'] = target_time['Power(mW)'].iloc[0]

                    pairs.append(pair_data)

        pairs_df = pd.DataFrame(pairs)
        return calculate_time_features(pairs_df)

    def prepare_test_data(template_path: str, station_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        template = pd.read_csv(template_path)
        test_pairs = []

        for _, row in tqdm(template.iterrows(), desc='Processing test cases'):
            seq_num = str(row['序號'])
            timestamp = pd.to_datetime(seq_num[:12], format='%Y%m%d%H%M')
            target_loc = int(seq_num[12:14])

            available_data = {}
            for loc, df in station_data.items():
                if timestamp in df['DateTime'].values:
                    available_data[loc] = df[df['DateTime'] == timestamp]

            for source_loc, source_df in available_data.items():
                if source_loc != target_loc:
                    test_pairs.append({
                        'DateTime': timestamp,
                        'target_location': target_loc,
                        'source_location': source_loc,
                        'l2_wind': source_df['WindSpeed(m/s)'].iloc[0],
                        'l2_pressure': source_df['Pressure(hpa)'].iloc[0],
                        'l2_temperature': source_df['Temperature(°C)'].iloc[0],
                        'l2_humidity': source_df['Humidity(%)'].iloc[0],
                        'l2_sunlight': source_df['Sunlight(Lux)'].iloc[0],
                        'l2_source_power': source_df['Power(mW)'].iloc[0],
                    })

        test_df = pd.DataFrame(test_pairs)
        return calculate_time_features(test_df)

    # Load all station data
    print("Loading station data...")
    station_data = {}
    for i in range(1, 18):
        file_path = f'{cleaned_data_path}/L{i}_Train_cleaned.csv'
        station_data[i] = load_station_data(file_path)

    # Create datasets
    print("\nCreating training dataset...")
    train_df = create_station_pairs(station_data, is_training=True)

    print("\nCreating test dataset...")
    test_df = prepare_test_data(upload_template_path, station_data)

    return train_df, test_df

if __name__ == "__main__":
    # Execute data preparation
    train_df, test_df = load_level2_data(
        cleaned_data_path='cleaned_data',
        upload_template_path='36_TestSet_SubmissionTemplate/upload.csv'
    )

    print("\nSaving processed datasets...")
    train_df.to_csv('l2_train_data.csv', index=False)
    test_df.to_csv('l2_test_data.csv', index=False)

    print("\nDataset shapes:")
    print(f"Training: {train_df.shape}")
    print(f"Testing: {test_df.shape}")

    print("\nFeature columns:")
    print(train_df.columns.tolist())