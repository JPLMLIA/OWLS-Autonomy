# DOMINE Setup Guide
Feel free to email [moorjani@jpl.nasa.gov](mailto:moorjani@jpl.nasa.gov) or
[eshaan@berkeley.edu](mailto:eshaan@berkeley.edu) for any questions.


## Environment Setup
[DOMINE Reference](https://github-fn.jpl.nasa.gov/COSMIC/COSMIC_DOMINE/blob/dev/README.md)
1. Create a conda environment with Python 2.7
   <br>
   `conda create -n domine-py2.7 python=2.7`
   <br>
   `conda activate domine-py2.7`
   
2. Manually set pip version to 18.1
   <br>
   `conda install pip=18.1`
   
3. **Clone** the DOMINE repository. 
   <br>
    `git clone https://github-fn.jpl.nasa.gov/COSMIC/COSMIC_DOMINE.git`
   <br>
   `cd COSMIC_DOMINE`
   
4. Install requirements for DOMINE. Either install requirements as:
    * `pip install --process-dependency-links .` or
    * `pip install -e .` if you plan to update DOMINE code.
    
5. Confirm that DOMINE is installed by running Python and calling `import domine`

## Data Pre-Processing
DOMINE requires certain modifications to your input dataset. In this guide, I will be discussing
how to setup DOMINE with a CSV file input. DOMINE documentation on creating custom dataloaders 
or using a NPZ or H5 dataloader is avaliable.
[here](https://github-fn.jpl.nasa.gov/COSMIC/COSMIC_DOMINE/wiki/DOMINE-Default-Loader-Schema).

DOMINE requires two columns, ids and groups, at index 0 and 1 and the ground truth to be the last column. 

If your data does not have ids and groups, DOMINE recommends you generate them as: 
```python
ids      = np.arange(#examples)  
groupids = np.arange(#examples)
```

Then, modify your dataset to make it DOMINE-ready with the following code: 
```python
# Construct list of headers for csv  
headers = ['ids', 'groups'] + featuress.tolist() + ['labels']
# Properly format string of comma-seperated header values  
headers = str(headers)[2:-2].replace("', '", ",")  

# Combine data to construct csv writable numpy matrix  
# Please note the matrix transposes (.T)  
out_data = np.vstack((ids, groups))  
out_data = np.vstack((out_data, X.T))  
out_data = np.vstack((out_data, Y)).T  

# Save out dataset with proper headers and comma seperated  
np.savetxt('my_dataest.csv', out_data, header=headers, delimiter=',', fmt='%s')  
```
The code above does not work for a `use_test` option. For more information, view the 
[wiki](https://github-fn.jpl.nasa.gov/COSMIC/COSMIC_DOMINE/wiki/DOMINE-Default-Loader-Schema). 

## Configuring DOMINE

The most important part about DOMINE is setting up your config file.

Here is an example of a config file I ran.

`05_HELM_DOMINE_config.yml`
```yaml
datasets:
    - HELM_experiment_features_DOMINE_no_inf_features:
          analysis: {
              'data_dir': '/path/to/data/directory/',
              'view_dir': '/path/to/data/directory/views/'
          }
parameters:
    CV_type: 'sklearn.model_selection.GroupKFold'
    CV_params: {
        n_splits: 5
    } 
    nested_CV_folds: 3

classifiers:
    - sklearn.neighbors.KNeighborsClassifier:
        params: {
            'n_neighbors': [3, 20],
            'algorithm': 'auto',
        }
        search: {
            'method': 'grid',
            'distribution': 'uniform',
            'n_samples': 150
        }
```

DOMINE operates with a server and a client.

In a bash window, run: `domine_server /path/to/config/file.yml /path/to/data/directory/`
<br>
In another bash window, run: `domine_client`.

Now, you are training with DOMINE.

## Validating Metrics for a DOMINE Run

DOMINE lets us compute stats and plot results. In this guide, I'll be showing how to compute stats. 
For more information plotting results, look [here](https://github-fn.jpl.nasa.gov/COSMIC/COSMIC_DOMINE)

The `compute_stats` command will take all models that DOMINE ran on, and find the one that optimizes 
for a single metric. Thus, prior to running `compute_stats`, we must create a metrics file. 

Below is an example of a metrics file I used. 

`metrics.yml`
```yaml
metrics:
  [
    {'f1': {
      'average': 'weighted'
    }},
    {'precision': {
      'pos_label': 'motile'
    }},
    {'recall': {
      'pos_label': 'motile'
    }},
    {'accuracy' : {}}
  ]
```

Now, in a bash window, run: 
`compute_stats
/path/to/config/file.yml
/path/to/data/directory/
/path/to/metrics/file.yml
/path/to/compute_stats/output.txt`

Congratulations! You should now be able to run DOMINE.