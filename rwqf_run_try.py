from AIHABs import AIHABs
from matplotlib import pyplot as plt


if __name__ == '__main__':

    osm_id = "5936797"
    db_name = "postgres"
    
    OPENEO_BACKEND = "https://openeo.dataspace.copernicus.eu"
    clid = "clientid"
    clse = "client_secret_key"
    provider_id = "CDSE"

    aihabs = AIHABs()
    aihabs.provider_id = provider_id
    aihabs.client_id = clid
    aihabs.client_secret = clse
    aihabs.osm_id = osm_id
    aihabs.db_name = db_name
    aihabs.freq = 'D'
    aihabs.t_shift = 3
    gdf_imputed, gdf_smooth = aihabs.run_analyse()
    
    fig_list = [0, 7, 12, 33, 55]

    for i in range(len(fig_list)):
        try:
            plt.figure(figsize=(20, 6))
            plt.title(f'Reservoir no.: {osm_id}; Point no.: {fig_list[i]}')
            # plt.plot(df_full[i], 'o')
            plt.plot(gdf_imputed['feature_value'].where(gdf_imputed['PID'] == fig_list[i]), alpha=0.2)
            plt.plot(gdf_smooth['feature_value'].where(gdf_smooth['PID'] == fig_list[i]), alpha=0.8)
        except:
            pass

    plt.show()
    



