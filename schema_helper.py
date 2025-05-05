#!/usr/bin/env python3

import os
import geopandas as gpd
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_data_schema(data_dir):
    """Extract and return the schema of all data files."""
    schema = {}

    # Load parcels data
    parcels_path = os.path.join(data_dir, 'cambridge_parcels.geojson')
    if os.path.exists(parcels_path):
        try:
            parcels = gpd.read_file(parcels_path)
            schema['parcels'] = {
                'columns': parcels.columns.tolist(),
                'sample': parcels.iloc[0].to_dict() if len(parcels) > 0 else {},
                'path': parcels_path
            }
            logger.info(
                f"Loaded parcels data with {len(parcels.columns)} columns")
        except Exception as e:
            logger.error(f"Error loading parcels data: {e}")

    # Load POI data
    poi_path = os.path.join(data_dir, 'cambridge_poi_processed.geojson')
    if os.path.exists(poi_path):
        try:
            poi = gpd.read_file(poi_path)
            schema['poi'] = {
                'columns': poi.columns.tolist(),
                'sample': poi.iloc[0].to_dict() if len(poi) > 0 else {},
                'path': poi_path
            }
            logger.info(f"Loaded POI data with {len(poi.columns)} columns")
        except Exception as e:
            logger.error(f"Error loading POI data: {e}")

    # Load census data
    census_path = os.path.join(
        data_dir, 'cambridge_census_cambridge_pct.geojson')
    if os.path.exists(census_path):
        try:
            census = gpd.read_file(census_path)
            schema['census'] = {
                'columns': census.columns.tolist(),
                'sample': census.iloc[0].to_dict() if len(census) > 0 else {},
                'path': census_path
            }
            logger.info(
                f"Loaded census data with {len(census.columns)} columns")
        except Exception as e:
            logger.error(f"Error loading census data: {e}")

    # Load spending data
    spend_path = os.path.join(data_dir, 'cambridge_spend_processed.csv')
    if os.path.exists(spend_path):
        try:
            spend = pd.read_csv(spend_path)
            schema['spend'] = {
                'columns': spend.columns.tolist(),
                'sample': spend.iloc[0].to_dict() if len(spend) > 0 else {},
                'path': spend_path
            }
            logger.info(
                f"Loaded spending data with {len(spend.columns)} columns")
        except Exception as e:
            logger.error(f"Error loading spending data: {e}")

    # Define Harvard Square location (this is used in many examples)
    schema['harvard_square'] = {
        'coordinates': (-71.1189, 42.3736),
        'description': 'Fixed coordinates for Harvard Square (used in multiple examples)'
    }

    # Define subway stations (commonly used in examples)
    schema['subway_stations'] = {
        'locations': [
            (-71.1189, 42.3736),  # Harvard Square
            (-71.1031, 42.3656),  # Central Square
            (-71.0865, 42.3625),  # Kendall/MIT
            (-71.1226, 42.3782),  # Porter Square
            (-71.1429, 42.3954)   # Alewife
        ],
        'description': 'Fixed coordinates for subway stations'
    }

    # Define use code mappings (based on sample code analysis)
    schema['use_codes'] = {
        'commercial': ['300', '302', '316', '323', '324', '325', '326', '327',
                       '330', '332', '334', '340', '341', '343', '345', '346',
                       '353', '362', '375', '404', '406', '0340', '0406'],
        'retail': ['323', '324', '325', '326', '327', '330'],
        'vacant_commercial': ['390', '391', '392', '3922'],
        'residential': ['101', '1014', '102', '1028', '104', '105', '109',
                        '1094', '1095', '1098', '111', '112', '113', '114', '121',
                        '970', '9700', '9421'],
        'mixed_use': ['0101', '0104', '0105', '0111', '0112', '0121', '013',
                      '031', '0340', '0406', '041', '0942'],
        'industrial': ['400', '401', '407', '413'],
        'office': ['340', '343', '345', '346', '0340', '404', '406', '0406']
    }

    # Extract sample code from one of the test samples
    sample_path = os.path.join(data_dir, 'spatial_samples.json')
    if os.path.exists(sample_path):
        try:
            with open(sample_path, 'r') as f:
                samples = json.load(f)
            if samples and len(samples) > 0:
                schema['sample_code'] = samples[0].get('Code', '')
                logger.info("Loaded sample code for reference")
        except Exception as e:
            logger.error(f"Error loading sample code: {e}")

    return schema


# If run directly, print the schema
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract data schema information")
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Directory containing data files")
    args = parser.parse_args()

    schema = get_data_schema(args.data_dir)

    # Print schema information for quick reference
    for dataset, info in schema.items():
        if dataset == 'use_codes':
            print(f"\n{dataset.upper()}:")
            for category, codes in info.items():
                print(f"  {category}: {len(codes)} codes")
        elif dataset not in ['sample_code', 'harvard_square', 'subway_stations']:
            print(f"\n{dataset.upper()} COLUMNS:")
            if 'columns' in info:
                print(", ".join(info['columns']))

    # Save full schema to file
    with open('data_schema.json', 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"\nFull schema saved to data_schema.json")
