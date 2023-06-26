#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import numpy as np
import os


from dateutil.relativedelta import relativedelta


def read_dataframe(filename: str):
    
    categorical = ['PULocationID', 'DOLocationID']
    
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    df['ride_id'] = f'{2022:04d}/{2:02d}_' + df.index.astype('str')
    
    return df


def prepare_dictionaries(df: pd.DataFrame):
    
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    
    return dicts


def load_model(model_file: str):
    
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
        
    return dv, model


def save_results(df, y_pred, output_file):
    
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def apply_model(input_file, output_file, year, month):
    

    print(f'reading the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f'loading the model...')
    model_file = 'model.bin' 
    dv, model = load_model(model_file)

    X_val = dv.transform(dicts)
    

    print(f'applying the model...')
    y_pred = model.predict(X_val)
    
    print(y_pred.mean())

    print(f'saving the result to {output_file}...')
    save_results(df, y_pred, output_file)

    return output_file


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=' predict taxi ride durations.')
    parser.add_argument('year', type=int, help='Year of the input data')
    parser.add_argument('month', type=int, help='Month of the input data')
    args = parser.parse_args()

    year = args.year
    month = args.month

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    output_file = f'output/predictions_{year}-{month}.parquet'

    apply_model(input_file, output_file, year, month)
