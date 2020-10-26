## Config Parameters (as they appear in config) ##
* tracker_settings:
  * skip_frames: int
  * diff_comp:
     * median_window: int
     * absthresh: float
     * pcthresh: float
  * clustering:
     * dbscan:
       * epsilon_px: float
       * min_weight: float
     * filters:
        * min_px: int
  * tracking:
     * allow_cluster_aggregation: bool
     * use_acceleration: bool
     * max_assignment_dist: float
     * max_projected_frames: int
     * min_track_obs: int

# The Tracker Algorithm #

The tracker algorithm follows three main steps. At each frame:
1. Compute a processed/filtered diff image
2. Run clustering on the diff image
3. Assign clusters to current tracks/create new tracks

## Diff Image ##

### Compute rolling median ###

We first compute a median image over the past `median_window` frames. Median is computed independently per pixel using np.median(). A higher value is more robust to noise / particles passing through, a lower value is more adaptive (say, to a large out of focus particle moving slowly). This becomes less efficient both in memory and runtime as `median_window` increases. Perf could be improved by leveraging that the window only changes by one addition and one subtraction each frame. Thus the median can shift by at most one index in a sorted list.

### Subtract median image from current frame ###

The goal of the median image is to model the current background. By subtracting this image from the current frame we hope to highlight moving particles. The absolute value is taken to capture both bright and dark anomalies. Depending on particle speed and `median_window`, anywhere from just the leading edge, to particle body, to trailing edge can be captured. 

### Filter resulting diff ###

Even after median subtraction, the diff image is still noisy. To account for this noise and reduce complexity of the later clustering, we set both absolute and percentile thresholds on the diff. The absolute thresholding masks any diff pixels which fall below a raw pixel difference `absthresh`. The percent thresholding masks the smallest `pcthresh` percent of diff pixels. In this implementation, the percentile cutoff is precomputed and we then apply a single threshold using the max of the absolute and percentile thresholds.

Small note: The thresholds are used in tandem as they are robust to different variations in data. The absolute threshold accounts for varying point density. It aims to set a cutoff on the raw intensity difference between noise to noise and noise to particle. The percent threshold doesn't care about the magnitude of this raw difference; rather it is an apriori estimation on the density of particles as a percentage of pixels in the image.

### Percentile transformation ###

As a last step post filtering and before clustering, the diff image is normalized across the entire 0 to 255 8-bit range. All non-zero pixels are first sorted, then each pixel is assigned a value in the range 0-255 based linearly on its position in the sorted list. That is, the dimmest (non-zero) pixel in the diff is set to 0, the brightest is 255, and the median is floor(255/2). If multiple pixels tie in the sorted list, they are all set to the same value (the mean of their would-be assignments had there been no tie).

No tests have been done without this normalization by the new team. While it is expected this normalization helps ensure the same clustering parameters work for every frame, it is worth testing without normalization to verify that assumption.

## Clustering ##

We run weighted DBSCAN out of the box on the filtered, normalized diff image. DBSCAN takes parameters `epsilon_px` - the "epsilon" DBSCAN parameter in units of pixels, and `min_weight` - the minimum sum of point weights within an epsilon ball to start a cluster.

Post-clustering, we then filter out any cluster that has fewer than `min_px` points (unweighted).

Each cluster is then reduced to a single center - the unweighted mean of the points. One obvious alternative would be to use the centroid of the weighted points (unimplemented as of now).

Lastly, clustering (and later tracking) is only run after `skip_frames` frames have passed. This is to allow sufficient frames for the median computation before attempting later tracker stages.

## Assign Clusters to Tracks ##

After clustering, each cluster point (center) is assigned to a current or new track. At frame one, each cluster is considered its own track. At subsequent frames, all current tracks are projected to get an estimate on the current position of the tracked particle. 

This projection is either a simple (position + estimated velocity * elapsed time) or additionally includes acceleration depending on the value of `use_acceleration`. Estimated velocity is computed using most recent two frames. Acceleration uses three. At this time it seems that acceleration estimation is too noisy to aid in the projection.

Each detected cluster is assigned to the closest projected track, so long as this distance is less than `max_assignment_dist` in pixels. If the cluster is not within `max_assignment_dist` of any track, it starts a new track.

Multiple clusters may be assigned to the same track. In this case either the closest cluster is assigned to the track (and the others discarded; they do not form new tracks) or the unweighted mean of the assigned clusters is used. Which behavior is controlled by `allow_cluster_aggregation`.

If a track has no assigned clusters, then it is considered unmatched for that frame. A track can remain unmatched for up to `max_projected_frames` before termination. During this time the track continues to be projected using its last known estimates, and at any time may be picked up again by a new matched cluster. These unmatched frames are currently set to NULL in the track output, but could be interpolated in the future.

Note that the `max_assignment_dist` parameter needs to strike a careful balance between allowing *some* error to avoid fragmented tracks, whilst preventing interference from neighboring tracks and/or noise. Setting `allow_cluster_aggregation` to false should help with this problem -> only considering closest means tracks aren't affected by additional clusters at the outskirts of the range. Additionally if two particles get close, tracks are correct as long as each cluster remains closest to its projected track. Of course, severe overlap will result in two particles forming a single cluster. Here we hope that the track projection can regain tracking after the particles separate.

All finished tracks (either terminated after being unmatched for too many frames, or the last experiment frame is hit) are trimmed of any trailing NULL's. If tracks are too short (fewer than `min_track_obs` non-NULL frames), they are omitted. The tracker algorithm then outputs the surviving tracks.
