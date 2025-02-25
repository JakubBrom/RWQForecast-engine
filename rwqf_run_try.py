from AIHABs import AIHABs


if __name__ == '__main__':

    osm_id = "15444638"
    db_name = "postgres"

    aihabs = AIHABs()
    aihabs.osm_id = osm_id
    aihabs.db_name = db_name
    aihabs.freq = 'D'
    aihabs.t_shift = 3
    gdf_imputed, gdf_smooth = aihabs.run_analyse()


