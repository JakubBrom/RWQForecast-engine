import os.path

from sqlalchemy import create_engine, exc, text
import datetime
from warnings import warn
import dill
import base64
import uuid

def add_model_to_table(table_name, db_name, user, feature, osm_id, orig_date, author, test_accuracy, pkl_file,
                       default=False):
    """
    The function for adding Pickle model to the database

    :param table_name: Name of the db table
    :param db_name: Database name
    :param user: User name
    :param feature: Water quality feature (ChlA, ChlB, PC, TSS, APC, PE, CX)
    :param osm_id: OSM object id (default = None)
    :param orig_date: Date of the model creation/uploading
    :param author: Name of the model author
    :param test_accuracy: Accuracy of the model
    :param pkl_file: Path to the pickle file
    :param default: Is the model default (default = False)
    :return:
    """

    # Connect to PostGIS
    engine = create_engine('postgresql://{user}@/{db_name}'.format(user=user, db_name=db_name))

    # Check if table exists
    query = text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '{tab_name}')".format(tab_name=table_name))

    with engine.connect() as connection:
        result = connection.execute(query)
        exists = result.scalar()

    if not exists:
        warn(f"The table does not exist in the database. The table {table_name} will be created.", stacklevel=2)
        # Create table
        # model_name, OSM_ID, feature, date, default, author, test_accuracy
        sql_query = text("CREATE TABLE IF NOT EXISTS {db_table} (id serial PRIMARY KEY, model_id text,osm_id text, date date, "
                         "feature varchar(50), model_name varchar(50), author varchar(50), test_accuracy double "
                         "precision, is_default boolean, pkl_file bytea)".format(db_table=table_name))

        with engine.connect() as connection:
            connection.execute(sql_query)
            connection.commit()

    # Add values to the table
    model_id = str(uuid.uuid4())
    model_name = os.path.splitext(os.path.basename(pkl_file))[0]
    orig_date = datetime.datetime.strptime(orig_date, "%Y-%m-%d").date()

    with open(pkl_file, 'rb') as f:
        pkl_model = dill.load(f)

    pkl_model = dill.dumps(pkl_model)       # convert to bytes
    pkl_model = base64.b64encode(pkl_model).decode('utf-8')


    if default:
        def_query = text("UPDATE {db_table} SET is_default = False WHERE feature = '{feature}'".format(
            db_table=table_name, feature=feature))

        with engine.connect() as connection:
            connection.execute(def_query)
            connection.commit()

    add_data_query = text("INSERT INTO {db_table} (model_id, osm_id, date, feature, model_name, author, test_accuracy, pkl_file, "
                   "is_default) VALUES ('{model_id}', '{osm_id}', '{date}', '{feature}', '{model_name}', '{author}', '{test_accuracy}', "
                      "'{pkl_file}', {default})".format(db_table=table_name, model_id=model_id,osm_id=osm_id, date=orig_date, feature=feature,
                                             model_name=model_name, author=author, test_accuracy=test_accuracy,
                                             pkl_file=pkl_model, default=default))

    with engine.connect() as connection:
        connection.execute(add_data_query)
        connection.commit()

    connection.close()
    engine.dispose()

    return


if __name__ == '__main__':

    db_name = "postgres"
    user = "postgres"
    tab_name = 'models_table'

    osm_id = None
    feature = 'ChlA'                   # ChlA, ChlB, PC, TSS, APC, PE, CX
    orig_date = "2024-08-23"
    default = True
    author = 'Jakub'
    test_accuracy = 0.87

    # Path to pkl file
    pkl_file = os.path.join(os.path.dirname(__file__), 'pomocne', 'modely', 'AI_model_test_3.pkl')

    # Run the function
    add_model_to_table(tab_name, db_name, user, feature, osm_id, orig_date, author, test_accuracy, pkl_file, default)
