import sys
import unittest
import geopandas as gpd
from pathlib import Path
import os
from sqlalchemy import create_engine, text

cwd = os.getcwd()
sys.path.append(cwd)

from get_random_points import point_mesh, get_sampling_points, get_vertices, generate_points_in_polygon
from matplotlib import pyplot as plt

class MyTestCase(unittest.TestCase):

    osm_id = 11092220
    db_name = "postgres"
    user = "postgres"
    db_table_reservoirs = "reservoirs"
    db_table_points = "test_selpoints"
    engine = create_engine('postgresql://{user}@/{db_name}'.format(user=user, db_name=db_name))
    sql_query = "SELECT * FROM {db_table} WHERE osm_id = '{osm_id}'".format(osm_id=str(osm_id), db_table=db_table_reservoirs)
    gdf = gpd.read_postgis(sql_query, engine, geom_col='geometry')
    reserv_polygon = gpd.GeoDataFrame(gdf, geometry='geometry', crs='epsg:4326')


    # def test_point_mesh(self):

    #     current_dir = Path(__file__).parent
    #     project_dir = current_dir.parent
    #     data_dir = project_dir / 'data_test'

    #     # Získání cesty k souboru 'soubor.txt' v adresáři 'data'
    #     polygon_path = data_dir / 'Vanern_wgs.gpkg'

    #     polygon = gpd.read_file(polygon_path)
    #     points = point_mesh(polygon)

    #     ax = polygon.plot(color='lightgray', figsize=(8, 8))
    #     points.plot(ax=ax, alpha=0.5, color='orange', edgecolor='black')
    #     plt.title("Random points")
    #     plt.show()

    #     return

    # def test_get_sampling_points(self):
    #     points, reserv = get_sampling_points(self.osm_id, self.db_name, self.user, self.db_table_reservoirs, self.db_table_points)
    #     # Plot the points and reservoir
    #     ax = reserv.plot(color='lightgray', figsize=(8, 8))
    #     points.plot(ax=ax, alpha=0.5, color='orange', edgecolor='black')
    #     plt.title("Random points")
    #     plt.show()

    # def test_get_vertices(self):

    #     polygon_geom = self.reserv_polygon['geometry'][0].simplify(0.0001, preserve_topology=True)
    #     vertices = get_vertices(polygon_geom)
    #     print(vertices)
    #     # Plot the points and reservoir
    #     plt.plot(vertices)
    #     plt.title("Random points")
    #     plt.show()

    def test_generate_points(self):
        points_clipped, points_selected, buff = generate_points_in_polygon(self.reserv_polygon, lake_buffer=-20, n_points_km=100, n_max_points=10000)
        ax = self.reserv_polygon.plot(color='lightgray', figsize=(8, 8))
        buff.plot(ax=ax, edgecolor="lightblue")
        # points_clipped.plot(ax=ax, alpha=0.5, color='orange', edgecolor='black')
        points_selected.plot(ax=ax, alpha=0.5, color='green', edgecolor='black')
        # centr.plot(ax=ax, alpha=0.5, color='orange', edgecolor='black')
        plt.title("Random points")
        plt.show()

if __name__ == '__main__':
    unittest.main()