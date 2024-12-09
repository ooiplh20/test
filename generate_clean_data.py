import pandas as pd
import os
from pathlib import Path

def merge_training_files(original_folder, additional_folder):
    """
    Merge training files from original and additional folders based on location identifier

    Args:
        original_folder: Path to folder containing original training files
        additional_folder: Path to folder containing additional training files
    Returns:
        dict: Dictionary with location IDs as keys and merged DataFrames as values
    """
    merged_data = {}

    # Process original training files first
    for file_name in os.listdir(original_folder):
        if file_name.endswith('_Train.csv'):
            # Extract location identifier (e.g., 'L2' from 'L2_Train.csv')
            location_id = file_name.split('_')[0]

            # Read the original file
            file_path = os.path.join(original_folder, file_name)
            df = pd.read_csv(file_path)
            merged_data[location_id] = df

    # Process and merge additional training files
    for file_name in os.listdir(additional_folder):
        if file_name.endswith('.csv'):
            # Extract location identifier (e.g., 'L2' from 'L2_Train_2.csv')
            location_id = file_name.split('_')[0]

            if location_id in merged_data:
                # Read the additional file
                file_path = os.path.join(additional_folder, file_name)
                additional_df = pd.read_csv(file_path)

                # Concatenate with existing data
                merged_data[location_id] = pd.concat(
                    [merged_data[location_id], additional_df],
                    ignore_index=True
                )

    # Sort and remove duplicates for each merged dataset
    for location_id in merged_data:
        df = merged_data[location_id]
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values('DateTime')
        df = df.drop_duplicates(subset=['DateTime'], keep='first')
        merged_data[location_id] = df

    return merged_data

def process_single_file(df):
    """
    Process a single DataFrame to:
    1. Resample data to 10-minute intervals
    2. Remove WindSpeed column
    3. Maintain proper datetime format for time series analysis

    Args:
        df: Input DataFrame
    Returns:
        DataFrame: Processed DataFrame
    """
    try:
        # Convert DateTime column to datetime type if it's not already
        df['DateTime'] = pd.to_datetime(df['DateTime'])

        # Set DateTime as index
        df = df.set_index('DateTime')

        # # Drop WindSpeed column if it exists
        # if 'WindSpeed' in df.columns:
        #     df = df.drop('WindSpeed', axis=1)

        # Resample to 10-minute intervals using mean
        df = df.resample('10min').mean()

        # Make sure the index has frequency information
        df.index = df.index.to_period('10min').to_timestamp('10min')

        # # Make round(2) for all columns except DateTime
        # for col in df.columns:
        #     if col != 'DateTime':
        #         df[col] = df[col].round(2)

        # # Remove WindSpeed column
        # if 'WindSpeed(m/s)' in df.columns:
        #     df = df.drop('WindSpeed(m/s)', axis=1)

        # Reset index to get DateTime back as a column
        df = df.reset_index()

        # Sort by DateTime to ensure proper time series order
        df = df.sort_values('DateTime')

        return df

    except Exception as e:
        print(f"Error processing DataFrame: {str(e)}")
        return None

def process_all_files(original_folder, additional_folder, output_folder):
    """
    Merge and process all training files, then save to output folder

    Args:
        original_folder: Path to folder containing original training files
        additional_folder: Path to folder containing additional training files
        output_folder: Path to save processed files
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Merge files first
    print("Merging files...")
    merged_data = merge_training_files(original_folder, additional_folder)

    # Process each merged dataset
    for location_id, df in merged_data.items():
        print(f"Processing {location_id} data...")
        processed_df = process_single_file(df)

        if processed_df is not None:
            # Create output filename
            output_file = os.path.join(output_folder, f"{location_id}_Train_cleaned.csv")

            # Save processed file
            processed_df.to_csv(output_file, index=False)
            print(f"Saved processed file to {output_file}")

# Usage example:
original_folder = "36_TrainingData"
additional_folder = "36_TrainingData_Additional_V2"
output_folder = "cleaned_data"

# Process all files
process_all_files(original_folder, additional_folder, output_folder)