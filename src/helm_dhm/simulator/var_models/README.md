# Models

### nov_2019_*.pickle
* Trained on 10 recordings from November 2019
  * 2019_11_11_14_35_15_328_medium_motile_DHM_No
  * 2019_11_11_14_49_58_387_medium_motile_DHM_No
  * 2019_11_12_09_26_26_655_medium_motile_DHM_No
  * 2019_11_12_09_59_14_007_medium_motile_DHM_No
  * 2019_11_12_10_02_19_811_sparse_motile_DHM_No
  * 2019_11_12_10_03_52_699_medium_motile_DHM_No
  * 2019_11_12_11_15_44_307_very-dense_motile_DHM_No
  * 2019_11_12_11_16_30_755_very-dense_motile_DHM_Yes
  * 2019_11_12_11_18_03_647_very-dense_motile_DHM_No
  * 2019_11_14_17_15_17_753_dense_non-motile_DHM_Yes
* No flow/drift present
* Organisms look to consist mostly of Chlamydamonas, but wasn't confirmed.
* The `motile` model was trained using all tracks hand labeled as motile. `nonmotile` model was trained on nonmotile ones.

### 2021_bsub*.pickle
* Trained on 2 recordings from November 2021
  * 2021_02_19_dhm_no_low_bsub_xx_xx_grayscale_lab_19
  * 2021_02_03_dhm_no_high_bsub_xx_xx_grayscale_lab_24
* No flow/drift present
* Live organisms are bsub
* The `motile` model was trained using all tracks hand labeled as motile. `nonmotile` model was trained on nonmotile ones.

### 2021_newport_wild*.pickle
* Trained on 14 recordings from Newport Beach in Apr. 2021
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_04
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_05
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_07
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_13
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_17
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_19
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_21
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_27
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_31
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_37
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_39
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_40
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_45
  *  2021_04_15_dhm_true_low_wild_xx_xx_grayscale_newport_53
* Flow is present (so you likely do not want to add more)
* Live organisms are wild and therefore unknown
* The `motile` model was trained using all tracks hand labeled as motile. `nonmotile` model was trained on nonmotile ones.