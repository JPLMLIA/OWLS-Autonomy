Joint Examination for Water-based Extant Life (JEWEL)
=====================================================

## Configuration

The JEWEL configuration file contains two top-level entries, `named_configs` for
named configurations, and `bin_configs` for defining per-bin configuration. The
named configurations provide a mechanism for re-using the same prioritization
configuration across bins. A special named configuration with the name `default`
will be used for any bin that does not have a configuration explicitly
specified. An example configuration file is located at
`src/cli/config/jewel_default.yml`.

### Named Configurations

Each named configuration should include two entries `prioritization_algorithms`
and `merging_algorithm`. The `prioritization_algorithms` section should include
the algorithm name and parameters for each instrument. Instruments without
prioritization algorithms defined will not have their data included for
downlink. The `merging_algorithm` should specify a single algorithm name and
parameters for merging data products across instruments after per-instrument
prioritization. Some algorithms do not require any parameters to be specified.

### Bin Configurations

A separate configuration can be specified for each bin, indexed by an integer
key representing the bin number. Data products are prioritized within each bin,
starting with the bin with the lowest number and proceeding to bins with higher
numbers. Products from higher-priority bins (with lower bin numbers) are
downlinked before products with lower priority (higher bin numbers).

Each bin configuration should specify a `prioritizer`, which is the name of one
of the named prioritization configurations from the `named_configs` section. In
addition to the prioritization configuration, there are three other optional
configurations for defining constraints. The `min_data_products` constraints
specifies the minimum number of Autonomous Science Data Products (ASDPs) that
should be included in the downlink from this bin. The `max_data_volume`
constraint specifies the maximum data volume to be downlinked from this bin.
Finally, if there is a conflict between these two constraints,
`constraint_precedence` (either "min" or "max") specifies which of the two
constraints should take precedence.

## Algorithm Options

### Prioritization Algorithms

The following algorithms can be used to prioritize ASDPs for individual
instruments within priority bins.

Since ASDP sizes can vary significantly across types while the initial SUE
values are bounded in the range `[0, 1]`, the SUE per byte values actually used
for prioritization can favor smaller ASDPs over larger ones. To overcome this, a
`base_utility" parameter is included for algorithms that use utility estimates
(Naive and MMR algorithms below). The final SUE is the initial SUE times the
base utility, creating a conversion factor to bring all utility values into the
same units (in terms of utility per byte) across instruments.

1. Naive Prioritizer (`naive`): use the initial SUE values (without any
   diversity descriptors) for prioritizing ASDPs.

   **Parameters**:
   - `base_utility`: the base utility value multiplied by the SUE to produce
       the final utility; see description above (default: 1.0)

2. FIFO Prioritizer (`fifo`): order the ASDPs by time stamp (ASDPs are
   downlinked in first-in-first-out, or FIFO, order).

   **Parameters**: _None_

3. MMR Prioritizer (`mmr`): order ASDPs using the Maximum Marginal Relevance
   (MMR) algorithm. This algorithm attempts to balance diversity and utility
   when selecting ASDPs for downlink. When deciding what to downlink next, it
   computes the distance of available products in diversity descriptor (DD)
   space to each ASDP that has already been downlinked or scheduled for
   downlink. It then discounts the initial SUE of each candidate ASDP using
   this distance such that experiments with similar content are less likely to
   be selected.

   A parameter `alpha` is used to determine the strength of this discount. If
   `alpha = 0`, no discounting is applied and the original SUE is kept. If
   `alpha = 1`, the final SUE is determined solely by diversity-discounted
   value. A value of `alpha = 0.5` equally balances both.

   **Parameters**:
   - `similarity_func`: the `name` and `parameters` of a valid similarity
       function (see Similarity Functions)
   - `alpha`: the `alpha` parameter of the MMR algorithm as described above
   - `base_utility`: the base utility value multiplied by the SUE to produce
       the final utility; see description above (default: 1.0)

### Similarity Functions

The following functions are options that can be used to compute similarity
scores for algorithms such as MMR that account for diversity when
prioritizing ASDPs.

1. Gaussian Similarity (`gaussian_similarity`): Computes a Gaussian similarity
   function using squared Euclidean distance between DDs.

   **Parameters**:
   - `scale_factor`: a multiplicative scale factor applied to the distance
       matrix (prior to squaring) in the exponent of the Gaussian similarity
       function

### Merging Algorithms

The following algorithms can be used to merge prioritized ASDPs across
instruments within a single bin.

1. Greedy Merge (`greedy_merger`): merges ASDPs across instruments by greedily
   selecting the ASDP with the largest `final_sue_per_byte` value.

   **Parameters**: _None_

2. Constrained Merge (`constraint_merger`): merges by selecting ASDPs with the
   largest `final_sue_per_byte` values subject to specified instrument-specific
   constraints.

   **Parameters**:
   - `constraints`: a dictionary of constraints for each instrument. For each
       instruments, `min_data_products`, `max_data_volume`, and
       `constraint_precedence` options can be specified, and these arguments
       have the same semantics as for Bin Configurations described above.
