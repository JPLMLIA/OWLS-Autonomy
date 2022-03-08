import pytest
from cli.ACME_flight_pipeline import *


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@pytest.mark.skip
def test_analyzer_2D_object():
    #check whether program crashes for unknown masses
    args = Namespace(data='data/ACME_test_data/500*.pickle', masses='../configs/compounds.yml', \
                     params='../configs/acme_config.yml', noplots=None, noexcel=None, debug_plots=True, \
                     saveheatmapdata=None, knowntraces=None, reprocess_version='test_v', cores=None,
                     reprocess_dir='data/ACME_test_data/lab_data/', outdir='data/ACME_test_data/output/')

    analyse_all_data(vars(args))
    #check that we have some output
    assert os.path.exists('data/ACME_test_data/output/500/Unknown_Masses/500_peaks.csv')

    # check whether program crashes for known masses
    args.knowntraces = True
    args.outdir = 'data/ACME_test_data/output/'
    analyse_all_data(vars(args))
    #check that we have some output
    assert os.path.exists('data/ACME_test_data/output/500/Known_Masses/500_peaks.csv')


    # test bulk reprocessing
    bulk_reprocess(vars(args))
    assert os.path.exists('data/ACME_test_data/lab_data/2019/500/reports/test_v/')