## Configuration File Construction

This README will continually evolve with more documentation on how to format a configuration file. Existing knowledge is below.

### Background subtraction
`detection: algorithm_settings: detection: minmax:`
I haven't gotten a handle on these, but they're used when looping through hologram images, computing rolling stats, and doing background subtraction. **It's not clear to me that the median subtracted frames are used when looking for pixel change -- the current difference calculations appears to happen on raw holograms.** We could swap this over to use the median subtracted frames, but the system may respond wildly to such a large change.

* lag
* absthresh - absolute threshold used when computing changed pixels. This and `pcthresh` are checked for which is larger before doing the threshold application
* pcthresh - percent threshold used when computing changed pixels. This and `absthresh` are checked for which is larger before doing the threshold application

### Point detection
Under `config: detection: algorithm_settings: helm_orig:`

* epsilon_px: higher vals makes this the DBSCAN clustering algorithm looser as it'll use a larger window when looking for changed pixels. Try some higher values like 5, 10, 20 to see if it more consistently detects particle locations. Larger values should help detect larger particles or particles that are more out of focus. This is an important one to tune
* min_samples: must reach this number of "changed" pixels within `epsilon_pix` to call that a particle detection. Lowering this from 512 to like 256 or 128 will loosen the filter. I imagine this will help detect smaller/slow moving particles where the absolute number of changed pixels is small
* threshold: pixel value difference that must be met to call a pixel "changed". Try lowering to loosen the particle detection filter. Lower values should allow more small/slow moving particles to be detected 

* min_px: Number of pixels that must be "core samples" in DBSCAN parlance. This is already pretty low, so I think it should be okay as is.
* max_uncert_px: I'm not sure exactly what this value does. It has something to do with the uncertainty of the major axis of an ellipsoid fitted to each proposed particle.

### Classifier params
`classifiers: MIL_Classifier:`

* up `n_estimators` at least to something like 50 or 100 (this is number of decision trees in the RF)
* On the other params, I would leave them be to start with. They are all doing data augmentation seemingly by simulating small changes to the track velocity/acceleration and then re-projecting that track. We'd need to talk to Gary to understand this better

### Track matching
`track: track_association_settings`

Used when matching true labeled tracks to auto-generated ones. I'm not sure if these will affect classification or not -- still looking more deeply.