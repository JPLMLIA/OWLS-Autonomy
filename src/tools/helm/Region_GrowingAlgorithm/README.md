# Overview
This is an experimental MHI-based tracking algorithm developed by Max Riekeles as part of his 2022 internship. It relies on seed points to set starting locations for analysis. The algorithm then tries to trace along pixels similar in color (where particles have passed).

Max Riekeles: riekeles@tu-berlin.de

## Components
* RegionGrowingAlgorithm1: uses clicks on individual pixels to determine seed points
* RegionGrowinAlgorithm2: uses a pre-defined grid to select seed points
* VisualizeTracks: Creates png files from the saved track (csv-files) from RegionGrowingAlgorithm2.
* Zprojections: Generates max-z projections from the reconstructed files. The max-z projections can then be used by HELM for generating the MHIs.
