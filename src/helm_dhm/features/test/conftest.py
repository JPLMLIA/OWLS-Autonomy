from helm_dhm.features import features
import pandas as pd
import pytest
import os
import json

def read_json_as_dict(file_name: str):
    with open(file_name, 'r') as f:
        return json.load(f)


TRACK_FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'data_track_features.csv')

@pytest.fixture
def data_track_features_df():
    return pd.read_csv(TRACK_FEATURES_PATH)


@pytest.fixture
def data_track_features_path():
    return TRACK_FEATURES_PATH


@pytest.fixture
def features_yml_file():
    return os.path.join(os.path.dirname(__file__), 'data', 'test_f.yml')


@pytest.fixture
def features_config_yml_file():
    return os.path.join(os.path.dirname(__file__), 'data', 'test_fc.yml')


@pytest.fixture
def hl_test_case():
    dataset_name = '2020_03_19_16_15_18_883_sparse_motile_simulated_comb'
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'verbose_hl_movieA.csv')), dataset_name


@pytest.fixture
def expected_extracted_features():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'data',
                                    'test_extract_features_from_dataset.csv'), index_col=0)


@pytest.fixture
def expected_functional_features():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'data',
                                    'test_features_functional.csv'))


@pytest.fixture
def mil_tracks():
    paths = [os.path.join(os.path.dirname(__file__), 'data', '000000023.track'),
             os.path.join(os.path.dirname(__file__), 'data', '000000068.track')]
    return list(map(lambda x: read_json_as_dict(x), paths))


@pytest.fixture
def expected_features_from_mil_tracks():
    return read_json_as_dict(os.path.join(os.path.dirname(__file__), 'data', 'expected_features_from_mil_tracks.json'))
