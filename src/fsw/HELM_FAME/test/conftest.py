import pytest
import os


@pytest.fixture
def create_partition_cases():
    base_path = os.path.join(os.path.dirname(__file__), 'data', 'test', 'test_sub')
    test_case = [os.path.join(base_path, 'case2', 'Holograms', '00001_holo.tif'),
                 os.path.join(base_path, 'case2', 'Holograms', '00002_holo.tif'),
                 os.path.join(base_path, 'case2', 'Holograms', '00003_holo.tif')]
    return test_case


@pytest.fixture
def find_dataset_cases():
    test_case = os.path.join(os.path.dirname(__file__), 'data', 'test')
    expected = [os.path.join(test_case, 'test_sub', 'case2'),
                os.path.join(test_case, 'test_sub', 'glob_test', 'case1'),
                os.path.join(test_case, 'test_sub', 'glob_test', 'deep_dir', 'case3')]
    expected = [path + os.sep for path in expected]
    return test_case, expected


@pytest.fixture
def get_basenames_and_files_cases():
    test_case = os.path.join(os.path.dirname(__file__), 'data', 'test')
    expected = (['case2', 'case1', 'case3'],
                [[os.path.join(os.path.dirname(__file__),'data', 'test', 'test_sub', 'case2', 'Holograms', '00001_holo.tif'),
                  os.path.join(os.path.dirname(__file__),'data', 'test', 'test_sub', 'case2', 'Holograms', '00002_holo.tif'),
                  os.path.join(os.path.dirname(__file__),'data', 'test', 'test_sub', 'case2', 'Holograms', '00003_holo.tif')],
                 [os.path.join(os.path.dirname(__file__),'data', 'test', 'test_sub', 'glob_test', 'case1', 'Holograms', '00001_holo.tif'),
                  os.path.join(os.path.dirname(__file__),'data', 'test', 'test_sub', 'glob_test', 'case1', 'Holograms', '00002_holo.tif'),
                  os.path.join(os.path.dirname(__file__),'data', 'test', 'test_sub', 'glob_test', 'case1', 'Holograms', '00003_holo.tif')],
                 [os.path.join(os.path.dirname(__file__),'data', 'test', 'test_sub', 'glob_test', 'deep_dir', 'case3', 'Holograms',
                               '00001_holo.tif'),
                  os.path.join(os.path.dirname(__file__),'data', 'test', 'test_sub', 'glob_test', 'deep_dir', 'case3', 'Holograms',
                               '00002_holo.tif'),
                  os.path.join(os.path.dirname(__file__),'data', 'test', 'test_sub', 'glob_test', 'deep_dir', 'case3', 'Holograms',
                               '00003_holo.tif')]], 0)
    return test_case, expected
