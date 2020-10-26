import pytest
import os
import tempfile

import pandas as pd

TRACK_FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'trackFeatures.csv')
TRACKER_REPO_PATH = os.path.join(os.path.dirname(__file__), 'data', 'return_valid_ids_test_dir')

@pytest.fixture
def trackFeatures():
    from tools.outlier_visualization.class_trackFeatures import TrackFeatures
    return TrackFeatures(TRACK_FEATURES_PATH)


@pytest.fixture
def unique_id_test_case():
    return 233

@pytest.fixture
def sort_trackfeatures_config_dict():
    config_dict = {
        'file_path': TRACK_FEATURES_PATH,
        'output_path': os.path.join(tempfile.mkdtemp(), 'temp.csv'),
        'motility': 'motile'
    }
    return config_dict

@pytest.fixture
def expected_sort_trackfeatures_dfs():
    file_paths = [os.path.join(os.path.dirname(__file__), 'data', 'sort_trackfeatures_main_motile.csv'),
                  os.path.join(os.path.dirname(__file__), 'data', 'sort_trackfeatures_main_nonmotile.csv'),
                  os.path.join(os.path.dirname(__file__), 'data', 'sort_trackfeatures_main_all.csv')]
    return tuple(map(lambda x: pd.read_csv(x), file_paths))

@pytest.fixture
def active_files_getter_test_case():
    test_case_1 = '2019.11.12_11.20.23.97'
    test_case_2 = '2019.11.12_11.27.04.236'
    return test_case_1, test_case_2

@pytest.fixture
def sort_df_row_by_outlier_expected():
    return os.path.join(os.path.dirname(__file__), 'data', 'sort_dataframe_rows_by_outlier_count_expected.csv')

@pytest.fixture
def labels_with_fiji_location():
    return os.path.join(TRACKER_REPO_PATH, 'hand_labels', 'labels_with_fiji')


@pytest.fixture
def tracker_repo_location():
    return TRACKER_REPO_PATH

@pytest.fixture
def sort_df_row_by_misclassification_expected():
    return os.path.join(os.path.dirname(__file__), 'data',
                        'sort_dataframe_rows_by_misclassification_count_expected.csv')

@pytest.fixture
def visualize_autotracks_location():
    return os.path.join(os.path.dirname(__file__), 'data', '2019.11.12_09.26.26.655')

@pytest.fixture
def hand_label_location():
    return os.path.join(os.path.dirname(__file__), 'data', 'return_valid_ids_test_dir',
                        'hand_labels', 'labels_with_fiji', 'Jun27_2019',
                        'verbose_2019.06.21_16.19.13_shewanella_sparse_rongsen.csv')

@pytest.fixture
def outlier_categorized_summary_config_dict():
    config_dict = {
        'file_path': TRACK_FEATURES_PATH,
        'tracker_repo': TRACKER_REPO_PATH,
        'output_path': tempfile.mkdtemp(),
        'classifier': 'svmPLY',
        'feature': 'meanVelocity'
    }
    return config_dict
