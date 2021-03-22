## LabelBox Upload Cheat Sheet:
```
1.) Generate data using this script
2.) scp folders generated in staging directory 
        to /home/wwwml/public/mlia/owls_labeling hosted on shiva.jpl.nasa.gov
3.) Give others permissions to view the files using the command:
        chmod -R o+rw owls_labeling/
        run from /home/wwwml/public/mlia/
4.) Add the json created in the staging directory to LabelBox at
        https://app.labelbox.com/data. The dataset will be given 
        the name of the JSON
5.) Attach the dataset to your project under project->settings->datasets
```

## LabelBox Download Cheat Sheet:
```
1.) Ensure you have a LabelBox API key.
        Instructions to get one are here:
        https://labelbox.com/docs/api/getting-started
2.) Download the label metadata from Labelbox.  These are found
        under project->export.  Keep json selected for format. This
        becomes the --label_metadata_file argument.
3.) Results will be placed in --experiment_dir folder. Future versions of
        this script can place them in the experiment subdir directly, but
        for now they are placed at the root of the experiment_dir folder 
```