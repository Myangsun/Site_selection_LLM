import os
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import tempfile
import time
import logging
from typing import List, Dict, Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpatialVisualizationHandler:
    """Handler for creating visualizations of spatial analysis results."""

    def __init__(self, data_dir: str):
        """
        Initialize the visualization handler.

        Args:
            data_dir: Directory containing the geospatial datasets
        """
        self.data_dir = data_dir

        # Set up data file paths
        self.data_files = {
            'parcels': os.path.join(data_dir, 'cambridge_parcels.geojson'),
            'poi': os.path.join(data_dir, 'cambridge_poi_processed.geojson'),
            'census': os.path.join(data_dir, 'cambridge_census_cambridge_pct.geojson'),
            'spend': os.path.join(data_dir, 'cambridge_spend_processed.csv')
        }

        # Create output directory for visualizations
        self.output_dir = os.path.join(
            tempfile.gettempdir(), "spatial_analysis")
        os.makedirs(self.output_dir, exist_ok=True)

        # Preload common datasets to avoid repeated loading
        self._parcels = None
        self._poi = None
        self._census = None

        logger.info(
            f"Visualization handler initialized with data dir: {data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    @property
    def parcels(self) -> gpd.GeoDataFrame:
        """Lazy-load and cache parcels data."""
        if self._parcels is None:
            try:
                self._parcels = gpd.read_file(self.data_files['parcels'])
                logger.info(
                    f"Loaded parcels data with {len(self._parcels)} features")
            except Exception as e:
                logger.error(f"Error loading parcels data: {e}")
                # Create empty dataframe as fallback
                self._parcels = gpd.GeoDataFrame()
        return self._parcels

    @property
    def poi(self) -> gpd.GeoDataFrame:
        """Lazy-load and cache POI data."""
        if self._poi is None:
            try:
                self._poi = gpd.read_file(self.data_files['poi'])
                logger.info(f"Loaded POI data with {len(self._poi)} features")
            except Exception as e:
                logger.error(f"Error loading POI data: {e}")
                # Create empty dataframe as fallback
                self._poi = gpd.GeoDataFrame()
        return self._poi

    @property
    def census(self) -> gpd.GeoDataFrame:
        """Lazy-load and cache census data."""
        if self._census is None:
            try:
                self._census = gpd.read_file(self.data_files['census'])
                logger.info(
                    f"Loaded census data with {len(self._census)} features")
            except Exception as e:
                logger.error(f"Error loading census data: {e}")
                # Create empty dataframe as fallback
                self._census = gpd.GeoDataFrame()
        return self._census

    def visualize_parcels(self,
                          parcel_ids: List[str],
                          title: str = None,
                          highlight_color: str = "red",
                          context_color: str = "lightgrey",
                          show_context: bool = True,
                          figsize: Tuple[int, int] = (10, 8),
                          dpi: int = 300,
                          add_basemap: bool = False) -> Optional[str]:
        """
        Create a visualization of selected parcels.

        Args:
            parcel_ids: List of parcel IDs (ml column values) to highlight
            title: Title for the visualization (default: auto-generated)
            highlight_color: Color for highlighted parcels
            context_color: Color for context parcels
            show_context: Whether to show all parcels as context
            figsize: Figure size as (width, height) in inches
            dpi: Resolution in dots per inch
            add_basemap: Whether to add an OpenStreetMap basemap (requires contextily)

        Returns:
            Path to the saved visualization image or None if failed
        """
        try:
            # Check if we have parcels to visualize
            if not parcel_ids:
                logger.warning("No parcel IDs provided for visualization")
                return None

            # Filter for the selected parcels
            result_parcels = self.parcels[self.parcels['ml'].isin(parcel_ids)]

            if len(result_parcels) == 0:
                logger.warning(
                    f"No parcels found with the provided IDs. Selected: {len(parcel_ids)}, Found: 0")
                return None

            # Create the visualization
            fig, ax = plt.subplots(figsize=figsize)

            # Plot all parcels as context if requested
            if show_context:
                self.parcels.plot(ax=ax, color=context_color,
                                  edgecolor='grey', alpha=0.3)

            # Plot result parcels (highlighted)
            result_parcels.plot(ax=ax, color=highlight_color,
                                edgecolor='black', alpha=0.7)

            # Add basemap if requested
            if add_basemap:
                try:
                    import contextily as ctx
                    # Convert to web mercator for contextily
                    ax = ctx.add_basemap(ax, crs=self.parcels.crs.to_string(
                    ), source=ctx.providers.OpenStreetMap.Mapnik)
                except ImportError:
                    logger.warning(
                        "contextily package not installed, skipping basemap")
                except Exception as e:
                    logger.warning(f"Error adding basemap: {e}")

            # Add title and labels
            if title is None:
                title = f"Selected Parcels ({len(result_parcels)} results)"
            plt.title(title, fontsize=16)
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)

            # Add a timestamp to avoid overwriting files
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(
                self.output_dir, f"parcels_{timestamp}.png")

            # Save the figure
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()

            logger.info(
                f"Saved visualization to {output_path} ({len(result_parcels)} parcels)")
            return output_path

        except Exception as e:
            logger.error(f"Error creating parcel visualization: {e}")
            return None

    def visualize_with_poi(self,
                           parcel_ids: List[str],
                           poi_type: Optional[str] = None,
                           buffer_distance: int = 500,
                           title: str = None) -> Optional[str]:
        """
        Create a visualization of selected parcels with nearby POIs.

        Args:
            parcel_ids: List of parcel IDs to highlight
            poi_type: Type of POI to show (None for all)
            buffer_distance: Buffer distance in meters
            title: Title for the visualization (default: auto-generated)

        Returns:
            Path to the saved visualization image or None if failed
        """
        try:
            # Check if we have parcels to visualize
            if not parcel_ids:
                logger.warning("No parcel IDs provided for visualization")
                return None

            # Filter for the selected parcels
            result_parcels = self.parcels[self.parcels['ml'].isin(parcel_ids)]

            if len(result_parcels) == 0:
                logger.warning(f"No parcels found with the provided IDs")
                return None

            # Convert to projected CRS for accurate buffering
            result_parcels_proj = result_parcels.to_crs(
                epsg=26986)  # Massachusetts state plane

            # Create buffer around parcels
            buffered_geom = result_parcels_proj.geometry.buffer(
                buffer_distance).unary_union
            buffered_gdf = gpd.GeoDataFrame(
                geometry=[buffered_geom], crs=result_parcels_proj.crs)

            # Convert back to original CRS
            buffered_gdf = buffered_gdf.to_crs(result_parcels.crs)

            # Filter POI data if requested
            filtered_poi = self.poi
            if poi_type:
                filtered_poi = filtered_poi[filtered_poi['business_type'] == poi_type]

            # Create the visualization
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot all parcels as context
            self.parcels.plot(ax=ax, color='lightgrey',
                              edgecolor='grey', alpha=0.2)

            # Plot buffer area
            buffered_gdf.plot(ax=ax, color='lightyellow',
                              edgecolor='none', alpha=0.3)

            # Plot result parcels (highlighted)
            result_parcels.plot(ax=ax, color='red',
                                edgecolor='black', alpha=0.7)

            # Plot POIs
            filtered_poi.plot(ax=ax, color='blue',
                              markersize=20, alpha=0.7, marker='o')

            # Add title and labels
            if title is None:
                poi_desc = f"'{poi_type}' POIs" if poi_type else "POIs"
                title = f"Selected Parcels with {poi_desc} within {buffer_distance}m"
            plt.title(title, fontsize=16)
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)

            # Add a timestamp to avoid overwriting files
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(
                self.output_dir, f"parcels_poi_{timestamp}.png")

            # Save the figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved POI visualization to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating POI visualization: {e}")
            return None

    def visualize_heatmap(self,
                          parcel_ids: List[str],
                          value_column: str,
                          data_source: str = 'census',
                          title: str = None,
                          cmap: str = 'viridis') -> Optional[str]:
        """
        Create a heatmap visualization based on census or other data.

        Args:
            parcel_ids: List of parcel IDs to highlight
            value_column: Column name to use for heatmap values
            data_source: Source data ('census', 'parcels', etc.)
            title: Title for the visualization
            cmap: Colormap to use

        Returns:
            Path to the saved visualization image or None if failed
        """
        try:
            # Check if we have parcels to visualize
            if not parcel_ids:
                logger.warning("No parcel IDs provided for visualization")
                return None

            # Filter for the selected parcels
            result_parcels = self.parcels[self.parcels['ml'].isin(parcel_ids)]

            # Get the data for the heatmap
            if data_source.lower() == 'census':
                data_gdf = self.census
            elif data_source.lower() == 'parcels':
                data_gdf = self.parcels
            else:
                logger.warning(f"Unknown data source: {data_source}")
                return None

            # Check if value column exists
            if value_column not in data_gdf.columns:
                logger.warning(
                    f"Column '{value_column}' not found in {data_source} data")
                available_columns = ', '.join(data_gdf.columns)
                logger.info(f"Available columns: {available_columns}")
                return None

            # Create the visualization
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot heatmap
            data_gdf.plot(column=value_column, cmap=cmap, linewidth=0.1,
                          ax=ax, edgecolor='black', legend=True, alpha=0.7)

            # Plot selected parcels on top
            result_parcels.plot(ax=ax, color='red',
                                edgecolor='black', alpha=0.5)

            # Add title and labels
            if title is None:
                title = f"Selected Parcels with {value_column} Heatmap"
            plt.title(title, fontsize=16)
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)

            # Add a timestamp to avoid overwriting files
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(
                self.output_dir, f"heatmap_{timestamp}.png")

            # Save the figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved heatmap visualization to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating heatmap visualization: {e}")
            return None

    def visualize_comparative(self,
                              result_sets: Dict[str, List[str]],
                              labels: Dict[str, str] = None,
                              colors: Dict[str, str] = None,
                              title: str = None) -> Optional[str]:
        """
        Create a comparative visualization of multiple sets of parcels.

        Args:
            result_sets: Dictionary mapping set IDs to lists of parcel IDs
            labels: Dictionary mapping set IDs to display labels
            colors: Dictionary mapping set IDs to colors
            title: Title for the visualization

        Returns:
            Path to the saved visualization image or None if failed
        """
        try:
            # Check if we have results to visualize
            if not result_sets:
                logger.warning("No result sets provided for visualization")
                return None

            # Create default labels and colors if not provided
            if labels is None:
                labels = {set_id: f"Set {i+1}" for i,
                          set_id in enumerate(result_sets.keys())}

            if colors is None:
                # Default color cycle
                default_colors = ['red', 'blue', 'green',
                                  'orange', 'purple', 'brown', 'pink', 'olive']
                colors = {set_id: default_colors[i % len(default_colors)]
                          for i, set_id in enumerate(result_sets.keys())}

            # Create the visualization
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot all parcels as context
            self.parcels.plot(ax=ax, color='lightgrey',
                              edgecolor='grey', alpha=0.2)

            # Plot each result set
            handles = []  # For legend
            for set_id, parcel_ids in result_sets.items():
                # Filter for the parcels in this set
                result_parcels = self.parcels[self.parcels['ml'].isin(
                    parcel_ids)]

                if len(result_parcels) == 0:
                    logger.warning(f"No parcels found for set {set_id}")
                    continue

                # Plot this result set
                color = colors.get(set_id, 'red')
                label = labels.get(set_id, str(set_id))

                # Plot and add to legend handles
                plot = result_parcels.plot(ax=ax, color=color, edgecolor='black',
                                           alpha=0.6, label=label)
                handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6,
                                             label=f"{label} ({len(result_parcels)} parcels)"))

            # Add legend
            ax.legend(handles=handles, loc='upper right')

            # Add title and labels
            if title is None:
                title = f"Comparison of Parcel Sets"
            plt.title(title, fontsize=16)
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)

            # Add a timestamp to avoid overwriting files
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(
                self.output_dir, f"comparison_{timestamp}.png")

            # Save the figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved comparative visualization to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating comparative visualization: {e}")
            return None
