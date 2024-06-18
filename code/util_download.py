# 
import os
import boto3
import aind_codeocean_api
import json
import boto3
import numpy as np
from aind_codeocean_api.codeocean import CodeOceanClient
from botocore.exceptions import ClientError
import shutil
import itertools
import glob

# datafolder =  '../results/'
def search_assets(query_asset='FIP_*'):    
    co_client = CodeOceanClient(domain=CO_DOMAIN, token=CO_TOKEN)
    response = co_client.search_all_data_assets(query="name:"+query_asset)
    data_assets_all = response.json()["results"]    
    data_assets = [r for r in data_assets_all if query_asset.strip('*') in r["name"]]
    return data_assets

def search_all_assets(query_asset='FIP_*', **kwargs):
    co_client = CodeOceanClient(domain=CO_DOMAIN, token=CO_TOKEN)
    has_more = True
    start = 0
    limit = 50    
    data_assets = []
    while has_more:
        response = co_client.search_data_assets(start=start, limit=limit, query='name:'+query_asset, **kwargs).json()
        has_more = response['has_more']
        results = response['results']
        data_assets_temp = [r for r in results if query_asset.strip('*') in r["name"]]
        data_assets = data_assets + data_assets_temp
        start += len(results)
    return data_assets

def download_assets(query_asset='FIP_*', query_asset_files='.csv',download_folder='../scratch/', max_assets_to_download=3):
    co_client = CodeOceanClient(domain=CO_DOMAIN, token=CO_TOKEN)
    response = co_client.search_all_data_assets(query="name:"+query_asset)
    data_assets_all = response.json()["results"]
    # Filter if data in asset name
    data_assets = [r for r in data_assets_all if query_asset.strip('*') in r["name"]]

    # Create s3 client
    s3_client = boto3.client('s3')
    s3_response = s3_client.list_buckets()
    s3_buckets = s3_response["Buckets"]

    for asset in data_assets[:max_assets_to_download]:
        # Get bucket id for datasets    
        dataset_bucket_prefix = asset['source_bucket']['bucket']
        asset_bucket = [r["Name"] for r in s3_buckets if dataset_bucket_prefix in r["Name"]][0]

        asset_name = asset["name"]
        asset_id = asset["id"]
        matching_string = query_asset_files

        response = s3_client.list_objects_v2(Bucket=asset_bucket, Prefix=asset_name)
        for object in response['Contents']:
            if (asset_name in object['Key']) and (matching_string in object['Key']):            
                filename = os.path.join(download_folder, object['Key'])
                pathname = os.path.dirname(filename)
                if not os.path.exists(pathname):
                    os.makedirs(pathname)
                # print('Downloading ' + filename)
                s3_client.download_file(asset_bucket, object['Key'], filename)

    s3_client.close()

def delete_downloaded_assets(query_asset='FIP_', query_asset_files='*.csv',download_folder='../scratch/', delete_folders=True):
    folders = [x[0] for x in os.walk(download_folder) if query_asset in x[0]]
    folders = np.sort(folders)[::-1]
    for folder in folders:
        if delete_folders:
            shutil.rmtree(folder)
        else:
            filenames = glob.glob(os.path.join(folder,query_asset_files), recursive=True)
            for filename in filenames:
                os.remove(filename)

def get_data_ids(data_asset):
    data_ids = data_asset['name'].strip('FIP_').split('_')
    return data_ids