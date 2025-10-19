"""
Mapillary Data Fetcher Module

This module provides functions to fetch Mapillary images within a bounding box
and convert them to GeoDataFrames for further analysis.
"""

import mapillary.interface as mly
from config import ACCESS_TOKEN
import json
import shapely.geometry
import geopandas as gpd
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import os
from typing import List, Dict, Union, Optional

# Set access token
mly.set_access_token(ACCESS_TOKEN)


def polygon_to_bbox_dict(polygon_coords: List[List[float]]) -> Dict[str, float]:
    """
    Convert polygon coordinates to a bounding box dictionary.
    
    :param polygon_coords: List of [longitude, latitude] coordinate pairs
    :type polygon_coords: List[List[float]]
    :return: Dictionary with west, south, east, north bounds
    :rtype: Dict[str, float]
    """
    lons = [pt[0] for pt in polygon_coords]
    lats = [pt[1] for pt in polygon_coords]
    return {
        "west": min(lons),
        "south": min(lats),
        "east": max(lons),
        "north": max(lats),
    }


def fetch_images_in_bbox(bbox: Union[Dict[str, float], List[List[float]]]) -> Dict:
    """
    Fetch Mapillary images within a bounding box.
    
    :param bbox: Either a bbox dictionary or polygon coordinates
    :type bbox: Union[Dict[str, float], List[List[float]]]
    :return: JSON data containing the images
    :rtype: Dict
    """
    if isinstance(bbox, list):
        bbox_dict = polygon_to_bbox_dict(bbox)
    else:
        bbox_dict = bbox
        
    data = json.loads(mly.images_in_bbox(bbox_dict))
    return data


def json_to_geodataframe(json_data: Dict) -> gpd.GeoDataFrame:
    """
    Convert JSON data to a GeoDataFrame.

    :param json_data: JSON data containing features with geometry
    :type json_data: Dict
    :return: GeoDataFrame containing the features
    :rtype: gpd.GeoDataFrame
    """
    features = json_data.get("features", [])
    geometries = []
    properties = []
    
    for feature in features:
        geometry = feature.get("geometry")
        properties.append(feature.get("properties", {}))
        
        if geometry:
            if geometry["type"] == "Point":
                geometries.append(shapely.geometry.Point(geometry["coordinates"]))
            elif geometry["type"] == "LineString":
                geometries.append(shapely.geometry.LineString(geometry["coordinates"]))
            elif geometry["type"] == "Polygon":
                geometries.append(shapely.geometry.Polygon(geometry["coordinates"][0]))
    
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
    return gdf


def fetch_and_convert_to_gdf(bbox: Union[Dict[str, float], List[List[float]]], 
                             output_path: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Fetch Mapillary images and convert directly to GeoDataFrame.
    
    :param bbox: Either a bbox dictionary or polygon coordinates
    :type bbox: Union[Dict[str, float], List[List[float]]]
    :param output_path: Optional path to save the GeoDataFrame. If exists, will skip fetching
    :type output_path: Optional[str]
    :return: GeoDataFrame containing the images
    :rtype: gpd.GeoDataFrame
    """
    # Check if output file already exists
    if output_path and os.path.exists(output_path):
        print(f"Datei {output_path} existiert bereits - wird übersprungen.")
        return gpd.read_file(output_path)
    
    print("Fetching images from Mapillary API...")
    data = fetch_images_in_bbox(bbox)
    gdf = json_to_geodataframe(data)
    
    # Save if output path is provided
    if output_path:
        save_gdf_to_file(gdf, output_path)
        print(f"GeoDataFrame saved to: {output_path}")
    
    return gdf


def save_gdf_to_file(gdf: gpd.GeoDataFrame, filepath: str, driver: str = 'GPKG') -> None:
    """
    Save GeoDataFrame to file.
    
    :param gdf: GeoDataFrame to save
    :type gdf: gpd.GeoDataFrame
    :param filepath: Path where to save the file
    :type filepath: str
    :param driver: File format driver (default: 'GPKG')
    :type driver: str
    """
    gdf.to_file(filepath, driver=driver)



def fetch_metadata_for_image(image_id: str, session: requests.Session, fields_str: str, 
                           access_token: str = ACCESS_TOKEN, max_retries: int = 3, 
                           delay: float = 0.001) -> Dict:
    """
    Fetch metadata for a single Mapillary image.
    
    :param image_id: The ID of the image to fetch metadata for
    :type image_id: str
    :param session: Requests session for connection pooling
    :type session: requests.Session
    :param fields_str: Comma-separated string of fields to fetch
    :type fields_str: str
    :param access_token: Mapillary API access token
    :type access_token: str
    :param max_retries: Maximum number of retry attempts
    :type max_retries: int
    :param delay: Delay between requests in seconds
    :type delay: float
    :return: Dictionary containing the image metadata
    :rtype: Dict
    """
    url = f"https://graph.mapillary.com/{image_id}?access_token={access_token}&fields={fields_str}"
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                time.sleep(delay)
                return r.json()
            else:
                print(f"Fehler bei {image_id}: Status {r.status_code}, Antwort: {r.text}")
        except Exception as e:
            print(f"Fehler bei {image_id} (Versuch {attempt+1}/{max_retries}): {e}")
        time.sleep(2 * (attempt + 1))
    time.sleep(delay)
    return {}


def fetch_metadata_batch(gdf: gpd.GeoDataFrame, output_path: str,
                        fields: Optional[List[str]] = None,
                        requests_per_minute: int = 50000,
                        safety_factor: float = 0.9,
                        failed_ids_path: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Fetch metadata for multiple Mapillary images efficiently using parallel processing.
    
    :param gdf: GeoDataFrame containing image IDs
    :type gdf: gpd.GeoDataFrame
    :param output_path: Path to save the merged GeoDataFrame with metadata
    :type output_path: str
    :param fields: List of metadata fields to fetch. If None, uses default fields
    :type fields: Optional[List[str]]
    :param requests_per_minute: Maximum requests per minute (API limit)
    :type requests_per_minute: int
    :param safety_factor: Safety factor to avoid hitting API limits (0.0-1.0)
    :type safety_factor: float
    :param failed_ids_path: Path to save failed image IDs. If None, uses default path
    :type failed_ids_path: Optional[str]
    :return: GeoDataFrame with merged metadata
    :rtype: gpd.GeoDataFrame
    """
    # Check if output file already exists
    if os.path.exists(output_path):
        print(f"Metadata-Datei {output_path} existiert bereits - Fetch Metadata wird übersprungen.")
        return gpd.read_file(output_path)
    
    if fields is None:
        fields = [
            "id", "altitude", "atomic_scale", "camera_parameters", "camera_type",
            "captured_at", "compass_angle", "computed_altitude", "computed_compass_angle",
            "computed_geometry", "computed_rotation", "creator", "exif_orientation",
            "geometry", "height", "make", "model", "thumb_256_url", "thumb_1024_url",
            "thumb_2048_url", "thumb_original_url", "merge_cc", "mesh", "sfm_cluster",
            "width", "detections"
        ]
    
    fields_str = ",".join(fields)
    
    # Calculate optimal threading parameters
    min_delay = 60.0 / requests_per_minute
    max_workers = max(1, int(requests_per_minute * safety_factor * min_delay))
    
    print(f"requests_per_minute: {requests_per_minute}, min_delay: {min_delay:.4f}s, max_workers: {max_workers}")
    
    # Prepare image IDs
    image_ids = gdf['id'].astype(str).tolist()
    
    results = []
    failed_ids = []
    
    # Set default failed_ids_path if not provided
    if failed_ids_path is None:
        base_dir = os.path.dirname(output_path) if output_path else os.getcwd()
        failed_ids_path = os.path.join(base_dir, "failed_ids.txt")
    
    # Estimate processing time
    estimated_time_min = (len(image_ids) * min_delay) / max_workers / 60
    print(f"Starte Download mit {max_workers} parallelen Threads für {len(image_ids)} Bilder (geschätzte Zeit: {estimated_time_min:.1f} Minuten)...")
    
    start_time = time.time()
    
    # Setup session with retry strategy
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    
    # Execute parallel requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(fetch_metadata_for_image, img_id, session, fields_str, ACCESS_TOKEN, 3, min_delay): img_id 
            for img_id in image_ids
        }
        
        for i, future in enumerate(tqdm(as_completed(future_to_id), total=len(image_ids), desc="API-Requests")):
            data = future.result()
            if data:
                results.append(data)
            else:
                failed_ids.append(future_to_id[future])
            
            if (i+1) % 50 == 0 or (i+1) == len(image_ids):
                elapsed = time.time() - start_time
    
    # Handle failed requests
    if failed_ids:
        print(f"{len(failed_ids)} IDs konnten nicht geladen werden. Siehe {failed_ids_path}")
        with open(failed_ids_path, "w") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")
    
    # Process results
    meta_df = pd.DataFrame(results)
    
    # Debug output
    print('meta_df.head():')
    print(meta_df.head())
    print('meta_df.columns:')
    print(meta_df.columns)
    
    # Filter and prepare data for merging
    if 'id' in meta_df.columns:
        meta_df = meta_df[meta_df['id'].notnull()]
    else:
        print('Warnung: Keine id-Spalte in den API-Ergebnissen!')
        return gdf
    
    # Ensure both ID columns are strings
    if 'id' in gdf.columns:
        gdf['id'] = gdf['id'].astype(str)
    if 'id' in meta_df.columns:
        meta_df['id'] = meta_df['id'].astype(str)
    
    # Merge and save
    if not meta_df.empty and 'id' in meta_df.columns:
        merged = gdf.merge(meta_df, on='id', how='left')
        merged.to_file(output_path, driver="GPKG")
        print(f"Merged data saved to: {output_path}")
        return merged
    else:
        print('Kein Merge möglich, da keine gültigen Metadaten vorhanden.')
        return gdf


def load_and_fetch_metadata(input_gpkg_path: str, output_gpkg_path: str,
                           fields: Optional[List[str]] = None,
                           limit: Optional[int] = None,
                           failed_ids_path: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load a GeoPackage with image IDs and fetch metadata for all images.
    
    :param input_gpkg_path: Path to input GeoPackage containing image IDs
    :type input_gpkg_path: str
    :param output_gpkg_path: Path to save output GeoPackage with metadata
    :type output_gpkg_path: str
    :param fields: List of metadata fields to fetch. If None, uses default fields
    :type fields: Optional[List[str]]
    :param limit: Limit number of images to process. If None, processes all
    :type limit: Optional[int]
    :param failed_ids_path: Path to save failed image IDs. If None, uses default path
    :type failed_ids_path: Optional[str]
    :return: GeoDataFrame with merged metadata
    :rtype: gpd.GeoDataFrame
    """
    # Check if output file already exists
    if os.path.exists(output_gpkg_path):
        print(f"Metadata-Datei {output_gpkg_path} existiert bereits - wird übersprungen.")
        return gpd.read_file(output_gpkg_path)
    
    # Load GeoPackage
    print(f"Loading GeoPackage from: {input_gpkg_path}")
    gdf = gpd.read_file(input_gpkg_path)
    if limit:
        gdf = gdf.head(limit)
        print(f"Limited to {limit} images")
    
    gdf.crs = "EPSG:4326"
    print(f"Loaded {len(gdf)} images")
    
    # Fetch metadata
    return fetch_metadata_batch(gdf, output_gpkg_path, fields, failed_ids_path=failed_ids_path)