import sys
from unittest import TestCase
import os
cwd = os.getcwd()
print(cwd)
sys.path.append(cwd) 
from get_random_points import get_sampling_points

from matplotlib import pyplot as plt
import geopandas as gpd

class Test(TestCase):

    osm_id = 5936797
    db_name = 'postgres'
    user = 'postgres'
    db_table_reservoirs = 'reservoirs'
    db_table_points = 'test_selpoints'
    
    def test_get_sampling_points(self):
        points, polygon = get_sampling_points(self.osm_id, self.db_name, self.user, self.db_table_reservoirs, self.db_table_points)
        
        # Plot the points and polygon
        fig, ax = plt.subplots()
        polygon.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black')
        points.plot(ax=ax, color='red', markersize=5)
        plt.title('Sampling Points and Polygon')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
