# Track-specific and movie-wide track features
- Extract and visualize track features
- Absolute (track specific) and relative (movie-wide) features are extracted
     - 18 track features:
     ```python
     ['mean_disp_x', 'mean_disp_y', 'mean_disp_angle', 'max_stepAngle', 'mean_stepAngle', 'autoCorr_stepAngle_lag1', 'autoCorr_stepAngle_lag2', 'track_length', 'max_speed', 'mean_speed', 'stdev_speed', 'autoCorr_speed_lag1', 'autoCorr_speed_lag2', 'max_accel', 'mean_accel', 'stdev_accel', 'autoCorr_accel_lag1', 'autoCorr_accel_lag2']
    ```
    - 3 relative track features: `['rel_theta_displacement', 'rel_dir_dot', 'rel_speed']`
- Features can be extracted from track data stored as CSV or .track formats
