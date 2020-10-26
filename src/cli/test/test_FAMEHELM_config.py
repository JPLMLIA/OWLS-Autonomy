import yaml

def _compare_keys(dict1, dict2, path=[]):

    missing = set(dict1).symmetric_difference(set(dict2))
    if len(missing):
        raise Exception("Unsynced configs found in " + str(path) + ": " + str(missing))
    else:
        # dict keys are the same but are there nested keys?
        for k in dict1.keys():
            if type(dict1[k]) is dict and type(dict2[k]) is dict:
                _compare_keys(dict1[k], dict2[k], path=path+[k])
            elif type(dict1[k]) is dict or type(dict2[k]) is dict:
                raise Exception("One of these is a dict but one isn't: " + str(path+[k]))

def test_config_sync():
    helm_config_path = "cli/configs/helm_config.yml"
    fame_config_path = "cli/configs/fame_config.yml"

    with open(helm_config_path, 'r') as h:
        helm_config = yaml.safe_load(h)
    
    with open(fame_config_path, 'r') as f:
        fame_config = yaml.safe_load(f)
    
    _compare_keys(helm_config, fame_config)