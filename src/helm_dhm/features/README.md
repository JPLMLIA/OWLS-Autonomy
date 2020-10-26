# Track-specific and movie-wide track features
- Extract and visualize track features
- Absolute (track specific) and relative (movie-wide) features are extracted
     - 18 absolute track features:
     ```python
     ["track_length", "max_velocity", "mean_velocity", "stdev_velocity", "autoCorr_vel_lag1", 
     "autoCorr_vel_lag2", "max_stepAngle", "mean_stepAngle", "autoCorr_stepAngle_lag1", 
     "autoCorr_stepAngle_lag2", "max_accel", "mean_accel", "stdev_accel", "autoCorr_accel_lag1", 
     "autoCorr_accel_lag2", "ud_x", "ud_y", "theta_displacement"]
    ```
    - 3 relative track features: `["rel_vel", "rel_theta_displacement", "rel_dir_dot"]`
- Scatter plots of feature vs feature can be created
- XY plots per single dataset are created to show all tracks of a movie in a single snapshot.
- Features can be extracted from track data stored as CSV or .track formats
