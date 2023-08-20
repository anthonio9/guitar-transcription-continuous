import configparser
import json
import os


config = configparser.ConfigParser()
config.read('config.ini')


def get_datasets_path():
    datasets_path_json = config['CUSTOM_PATH']['datasets_path']
    dataset_path_list = json.loads(datasets_path_json)

    return os.path.join(*dataset_path_list)


def get_models_path():
    models_path_json = config['CUSTOM_PATH']['models_path']
    models_path_list = json.loads(models_path_json)

    return os.path.join(*models_path_list)
