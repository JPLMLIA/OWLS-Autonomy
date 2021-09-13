"""
Implements the maximum marginal relevance (MMR) algorithm and related
functionality
"""
import numpy as np

from utils.logger import get_logger

from .diversity import load_similarity_func, gaussian_similarity

# Global variable for logging
logger = get_logger()

DEFAULT_SIMILARITY = gaussian_similarity()

# Note for developers: follow the example of `mmr` to implement additional
# prioritization algorithms, and `greedy_merger` to implement additional
# merging algorithms. The outer wrapper functions should take any necessary
# parameters and return a function that computes the prioritization lists using
# the same call signatures as the examples.
#
# If custom code is needed to parse parameters from the YAML configuration,
# define a function that takes a YAML dictionary containing configuration and
# returns a dictionary containing the loaded parameters, then assign this to
# the `load_params` attribute of the algorithm wrapper function. If the
# parameters can be passed in as-is to the algorithm wrapper function, then the
# `load_params` function does not need to be defined. For example, the
# `greedy_merger` algorithm does not take any parameters, let alone parameters
# with non-built-in types, so no corresponding `load_params` method is defined.
# On the other hand, `mmr` requires custom code for loading the appropriate
# similarity function.

def bin_prioritizer(prioritization_algorithms, merging_algorithm):
    """
    Returns a function that performs prioritization within a single bin using
    the provided per-type prioritization and merging algorithms.

    Parameters
    ----------
    prioritization_algorithms: dict
        a dict containing parameterized prioritization algorithms keyed by ASDP type
    merging_algorithm: callable
        a parameterized merging algorithm

    Returns
    -------
    prioritize: callable
        a function that takes a dict of per-type ASDPDB metadata, as returned by
        `asdpdb.load_asdp_metadata_by_type` and returns a list of dicts
        containing ordered ASDPs with metadata (in the format expected by
        `asdpdb.save_asdp_ordering`)
    """

    def prioritize(asdpdb_data):
        order_by_type = {}

        logger.info('Ordering ASDPs by type...')

        for asdp_type, metadata in asdpdb_data.items():
            if asdp_type not in prioritization_algorithms:
                logger.warning(
                    f'Skipping type {asdp_type} (no algorithm specified)'
                )
                continue

            logger.info(f'Ordering {asdp_type} ASDPs...')
            algorithm = prioritization_algorithms[asdp_type]
            order_by_type[asdp_type] = algorithm(
                asdp_type, metadata
            )

        logger.info('Merging products...')
        return merging_algorithm(order_by_type)

    return prioritize


def greedy_merger():
    """
    Returns an instance of the greedy merge algorithm; a single list is created
    by sorting all ASDPs by their `final_sue_per_byte` values

    Returns
    -------
    greedy_merge: callable
        an instance of the greedy merge algorithm
    """

    def greedy_merge(order_by_type):
        """
        Implements a greedy merge algorithm; a single list is created by
        sorting all ASDPs by their `final_sue_per_byte` values

        Parameters
        ----------
        order_by_type: dict
            a dict of lists, where each list contains dict corresponding to ASDPs
            ordered by priority

        Returns
        -------
        merged: list
            a single list containing the merged ASDP dicts
        """
        # Add all lists together
        all_products = sum(order_by_type.values(), [])

        # Sort and return combined list by final SUEs (per byte)
        sort_key = lambda p: -p['final_sue_per_byte']
        return sorted(all_products, key=sort_key)

    return greedy_merge


def fifo_merger():
    """
    Returns an instance of the FIFO merge algorithm; a single list is created
    by sorting all ASDPs by their `timestamp` values

    Returns
    -------
    fifo_merge: callable
        an instance of the fifo merge algorithm
    """

    def fifo_merge(order_by_type):
        """
        Implements a FIFO merge algorithm; a single list is created by
        sorting all ASDPs by their `timestamp` values

        Parameters
        ----------
        order_by_type: dict
            a dict of lists, where each list contains dict corresponding to ASDPs
            ordered by priority

        Returns
        -------
        merged: list
            a single list containing the merged ASDP dicts
        """
        # Add all lists together
        all_products = sum(order_by_type.values(), [])

        # Sort and return combined list by final SUEs (per byte)
        sort_key = lambda p: p['timestamp']
        return sorted(all_products, key=sort_key)

    return fifo_merge


def constraint_merger(constraints):
    """
    Returns an instance of the constrained greedy merge algorithm; a single
    list is created by sorting all ASDPs by their `final_sue_per_byte` values
    and enforcing instrument-specific constraints

    Parameters
    ----------
    constraints: dict
        a dictionary of constraints for each instrument

    Returns
    -------
    constraint_merge: callable
        an instance of the constraint merge algorithm
    """

    def constraint_merge(order_by_type):
        """
        Implements a constrained merge algorithm; a single list is created by
        sorting all ASDPs by their `final_sue_per_byte` values, subject to
        instrument-specific constraints

        Parameters
        ----------
        order_by_type: dict
            a dict of lists, where each list contains dict corresponding to ASDPs
            ordered by priority

        Returns
        -------
        merged: list
            a single list containing the merged ASDP dicts
        """
        # Add all lists together after applying constraints
        all_products = []

        for t, order in order_by_type.items():

            # Load config and populate with defaults if needed
            config = constraints.get(t, {})
            if 'max_data_volume' not in config:
                config['max_data_volume'] = np.inf
            if 'min_data_products' not in config:
                config['min_data_products'] = 0
            if 'constraint_precedence' not in config:
                config['constraint_precedence'] = 'min'

            selected_products = apply_constraints(order, config)
            n_prod = len(selected_products)
            logger.info(f'Retained {n_prod} {t} products during merge')

            all_products += selected_products

        # Sort and return combined list by final SUEs (per byte)
        sort_key = lambda p: -p['final_sue_per_byte']
        return sorted(all_products, key=sort_key)

    return constraint_merge


def naive(base_utility=1.0):
    """
    Returns a callable instance of the naive prioritization algorithm just
    using initial SUEs multiplied by the base utility

    Parameters
    ----------
    base_utility: float (default = 1.0)
        the base utility value multiplied by the SUE to produce the final
        utility

    Returns
    -------
    prioritize: callable
        a function that takes an ASDP type name and a dict of per-type ASDPDB
        metadata, as returned by `asdpdb.load_asdp_metadata_by_type`, and
        returns a list of dicts containing ordered ASDPs with metadata (in the
        format expected by `asdpdb.save_asdp_ordering`)
    """
    def prioritize(asdp_type, metadata):

        # Extract metadata entries
        ids = metadata['asdp_id']
        sue = metadata['sue']
        ts = metadata['timestamp']
        untransmitted = metadata['downlink_status']
        n_untransmitted = np.sum(untransmitted)

        if n_untransmitted == 0:
            logger.info(f'No untransmitted {asdp_type} products to prioritize')
            return []

        size_bytes = metadata['asdp_size_bytes']
        sue_per_byte = sue / size_bytes

        # Fill in bad values with zeros
        sue_per_byte[np.isnan(sue_per_byte)] = 0.0
        sue_per_byte[np.isinf(sue_per_byte)] = 0.0

        order = np.argsort(-sue_per_byte)

        for cand_id in order:
            if untransmitted[cand_id]:
                logger.info(
                    f'Selected ASDP {ids[cand_id]}, '
                    f'initial SUE = {sue_per_byte[cand_id]:.2e}'
                )

        products = [
            {
                'asdp_id': ids[cand_id],
                'initial_sue': sue[cand_id],
                'final_sue': base_utility * sue[cand_id],
                'initial_sue_per_byte': sue_per_byte[cand_id],
                'final_sue_per_byte': base_utility * sue_per_byte[cand_id],
                'size_bytes': size_bytes[cand_id],
                'timestamp': ts[cand_id],
            }
            for cand_id in order
            if untransmitted[cand_id]
        ]
        return products

    return prioritize


def fifo():
    """
    Returns a callable instance of the first-in-first-out (FIFO) prioritization
    algorithm that sorts ASDPs by timestamp

    Returns
    -------
    prioritize: callable
        a function that takes an ASDP type name and a dict of per-type ASDPDB
        metadata, as returned by `asdpdb.load_asdp_metadata_by_type`, and
        returns a list of dicts containing ordered ASDPs with metadata (in the
        format expected by `asdpdb.save_asdp_ordering`)
    """
    def prioritize(asdp_type, metadata):

        # Extract metadata entries
        ids = metadata['asdp_id']
        sue = metadata['sue']
        ts = metadata['timestamp']
        untransmitted = metadata['downlink_status']
        n_untransmitted = np.sum(untransmitted)

        if n_untransmitted == 0:
            logger.info(f'No untransmitted {asdp_type} products to prioritize')
            return []

        size_bytes = metadata['asdp_size_bytes']
        sue_per_byte = sue / size_bytes

        # Fill in bad values with zeros
        sue_per_byte[np.isnan(sue_per_byte)] = 0.0
        sue_per_byte[np.isinf(sue_per_byte)] = 0.0

        order = np.argsort(ts)

        for cand_id in order:
            if untransmitted[cand_id]:
                logger.info(
                    f'Selected ASDP {ids[cand_id]}, '
                    f'initial SUE = {sue_per_byte[cand_id]:.2e}'
                )

        products = [
            {
                'asdp_id': ids[cand_id],
                'initial_sue': sue[cand_id],
                'final_sue': sue[cand_id],
                'initial_sue_per_byte': sue_per_byte[cand_id],
                'final_sue_per_byte': sue_per_byte[cand_id],
                'size_bytes': size_bytes[cand_id],
                'timestamp': ts[cand_id],
            }
            for cand_id in order
            if untransmitted[cand_id]
        ]
        return products

    return prioritize


def mmr(alpha, base_utility=1.0, similarity_func=DEFAULT_SIMILARITY):
    """
    Returns a callable instance of the MMR algorithm.

    Parameters
    ----------
    alpha: float [0, 1]
        the fraction of the diversity-adjusted SUE that is used in convex
        combination with the original SUE
            0 := revert to the original SUE (ignores diversity)
            1 := completely rely on diversity-adjusted SUE
    base_utility: float (default = 1.0)
        the base utility value multiplied by the SUE to produce the final
        marginal utility
    similarity_func: callable
        a function to compute the similarity between entries (takes a
        single argument containing ASDP metadata)

    Returns
    -------
    prioritize: callable
        a function that takes an ASDP type name and a dict of per-type ASDPDB
        metadata, as returned by `asdpdb.load_asdp_metadata_by_type`, and
        returns a list of dicts containing ordered ASDPs with metadata (in the
        format expected by `asdpdb.save_asdp_ordering`)
    """

    def prioritize(asdp_type, metadata):

        # Extract metadata entries
        ids = metadata['asdp_id']
        sue = metadata['sue']
        ts = metadata['timestamp']
        untransmitted = metadata['downlink_status']
        n_untransmitted = np.sum(untransmitted)

        if n_untransmitted == 0:
            logger.info(f'No untransmitted {asdp_type} products to prioritize')
            return []

        candidates = set(np.where(untransmitted)[0])
        others = list(set(range(len(sue))) - candidates)

        # Compute similarity matrix
        similarity = similarity_func(metadata)

        # Compute per-byte utility
        size_bytes = metadata['asdp_size_bytes']
        sue_per_byte = sue / size_bytes

        # Fill in bad values with zeros
        sue_per_byte[np.isnan(sue_per_byte)] = 0.0
        sue_per_byte[np.isinf(sue_per_byte)] = 0.0

        logger.info(f'Prioritizing {n_untransmitted} {asdp_type} candidates')

        order = []
        marginal_sue = []
        while len(candidates) > 0:
            candidx = np.array(sorted(candidates))

            # Get maximum similarity with selected/transmitted candidates
            all_others = np.array(order + others)
            if len(all_others) > 0:
                sim_cand = similarity[candidx]
                sim_max = np.max(sim_cand[:, all_others], axis=1)
            else:
                sim_max = np.zeros(len(candidx))

            # Get SUEs for remaining candidates
            sue_cand = sue_per_byte[candidx]

            # Compute the MMR scores
            diversity_factors = (
                (1 - alpha) + (alpha * (1.0 - sim_max))
            )
            scores = sue_cand * diversity_factors * base_utility

            # Select the best candidate and update lists
            best_idx = np.argmax(scores)
            best = candidx[best_idx]
            marginal_sue.append(np.max(scores))
            order.append(best)
            candidates.remove(best)

            logger.info(
                f'Selected ASDP {ids[best]}, '
                f'initial SUE = {sue_per_byte[best]:.2e}, '
                f'marginal SUE = {marginal_sue[-1]:.2e}, '
                f'discount factor = {diversity_factors[best_idx]:.2f}'
            )

        products = []
        for cand_id, final_sue_per_byte in zip(order, marginal_sue):
            final_sue = (final_sue_per_byte * size_bytes[cand_id])
            products.append({
                'asdp_id': ids[cand_id],
                'initial_sue': sue[cand_id],
                'final_sue': final_sue,
                'initial_sue_per_byte': sue_per_byte[cand_id],
                'final_sue_per_byte': final_sue_per_byte,
                'size_bytes': size_bytes[cand_id],
                'timestamp': ts[cand_id],
            })
        return products

    return prioritize


def mmr_load_params(param_config):
    """
    Loads parameters for the MMR algorithm from YAML configuration

    Parameters
    ----------
    param_config: dict
        YAML config dict containing parameters for the MMR algorithm

    Returns
    -------
    params: dict
        a dictionary of loaded parameters for the MMR algorithm
    """
    params = {}
    for k, v in param_config.items():
        if k == 'similarity_func':
            similarity_func = load_similarity_func(v)
            if similarity_func is None:
                return
            params[k] = similarity_func
        else:
            params[k] = v

    return params


# Associate the MMR parameter loading code with the MMR algorithm (in an
# alternate universe, the MMR algorithm is a class, and the `load_params` is
# class method)
mmr.load_params = mmr_load_params


def prioritizer(bin_configs, named_configs):
    """
    Returns a function that performs prioritization across all data products

    Parameters
    ----------
    bin_configs: dict
        a dict containing parsed per-bin configuration
    named_configs: dict
        a dict containing parsed, named prioritizer configs

    Returns
    -------
    prioritize: callable
        a function that takes a dict of per-bin-and-type ASDPDB metadata, as
        returned by `asdpdb.load_asdp_metadata_by_bin_and_type` and returns a
        list of dicts containing ordered ASDPs with metadata (in the format
        expected by `asdpdb.save_asdp_ordering`)
    """

    def get_config(priority_bin):
        """
        Performs lookup of named configurations to load per-bin configuration;
        default values are used if not provided
        """
        bin_config = bin_configs.get(priority_bin, {})
        pri_config = bin_config.get('prioritizer', 'default')
        prioritizer = named_configs.get(pri_config, None)
        max_data_volume = bin_config.get('max_data_volume', np.inf)
        min_data_products = bin_config.get('min_data_products', 0)
        constraint_precedence = bin_config.get('constraint_precedence', 'min')
        return {
            'prioritizer': prioritizer,
            'max_data_volume': max_data_volume,
            'min_data_products': min_data_products,
            'constraint_precedence': constraint_precedence,
        }

    def prioritize(asdpdb_data):

        logger.info('Ordering ASDPs by bin...')

        all_products = []

        for pbin, asdpdb_by_type in sorted(asdpdb_data.items()):
            logger.info(f'Ordering bin {pbin} ASDPs...')

            cfg = get_config(pbin)
            prioritizer = cfg['prioritizer']
            if prioritizer is None:
                logger.warning(
                    f'Skipping bin {pbin} (no algorithm specified)'
                )
                continue

            bin_order = prioritizer(asdpdb_by_type)
            selected_products = apply_constraints(bin_order, cfg)
            all_products += selected_products

        return all_products

    return prioritize


def apply_constraints(ordering, config):
    """
    Applies data volume and cardinality constraints to a list of ordered
    products

    Parameters
    ----------
    ordering: list
        list of dicts containing ordered ASDPs with metadata (in the format
        expected by `asdpdb.save_asdp_ordering`)
    config: dict
        dictionary containing configuration with entries `min_data_products`,
        `max_data_volume`, and `constraint_precedence`

    Returns
    -------
    selected_products: list
        subset of the `ordering` list of products subject to the given
        constraints
    """

    cumulative_size = np.cumsum([
        o['size_bytes'] for o in ordering
    ])
    acceptable = np.nonzero(
        cumulative_size <= config['max_data_volume']
    )[0]
    upper = 0 if len(acceptable) == 0 else acceptable[-1] + 1
    lower = config['min_data_products']

    if (lower > upper) and (config['constraint_precedence'] == 'min'):
        # Lower bound overrides upper bound
        selected_products = ordering[:lower]
    else:
        # Otherwise, upper bound is used
        selected_products = ordering[:upper]

    return selected_products


def load_algorithm(config):
    """
    Loads a parameterized algorithm from a YAML config dict.

    Parameters
    ----------
    config: dict
        dictionary containing YAML configuration with an "algorithm" entry for
        a similarity function defined within the global namespace of this
        module, and (optionally) a "parameters" entry containing a dict of
        parameters required by that similarity function

    Returns
    -------
    algorithm: callable
        returns a parameterized ASDP type-specific prioritization algorithm
        corresponding to the provided configuration, or None if an error
        occurred during loading
    """

    alg_name = config.get('algorithm', None)
    if alg_name is None:
        logger.warning('No algorithm specified.')
        return

    # Look for the algorithm by name in the global namespace
    algorithm = globals().get(alg_name, None)
    if algorithm is None:
        logger.warning(f'Algorithm "{alg_name}" not found.')
        return

    # Get parameters from config and load them with the custom similarity
    # function/class code if defined
    param_config = config.get('parameters', {})
    if hasattr(algorithm, 'load_params'):
        params = algorithm.load_params(param_config)
    else:
        params = param_config

    if params is None:
        return

    return algorithm(**params)


def load_bin_prioritizer(config):
    """
    Loads a callable bin prioritizer from the provided YAML configuration

    Parameters
    ----------
    config: dict
        YAML configuration dict for the prioritizer containing
        `prioritization_algorithms` and `merging_algorithm` entries (see
        default JEWEL config file for an example)

    Returns
    -------
    bin_prioritizer: callable
        a prioritization function constructed by calling the `bin_prioritizer`
        function with the parameters parsed from the YAML configuration
    """

    # Load prioritization algorithms from config
    prioritization_algorithms_config = config.get(
        'prioritization_algorithms', {}
    )
    prioritization_algorithms = {}
    for asdp_type, c in prioritization_algorithms_config.items():
        alg = load_algorithm(c)
        if alg is None:
            logger.warning(f'Could not load algorithm for {asdp_type}')
            continue
        prioritization_algorithms[asdp_type] = alg

    # Load merging algorithm from config
    if 'merging_algorithm' not in config:
        logger.warning(
            'No merging algorithm specified; using greedy merger'
        )
        merging_algorithm = greedy_merger()
    else:
        merging_algorithm = load_algorithm(
            config['merging_algorithm']
        )
        if merging_algorithm is None:
            logger.warning(
                'Could not load merging algorithm; '
                'falling back to default.'
            )
            merging_algorithm = greedy_merger()

    # Instantiate and return prioritizer
    return bin_prioritizer(prioritization_algorithms, merging_algorithm)


def load_prioritizer(config):
    """
    Loads a callable prioritizer from the provided YAML configuration

    Parameters
    ----------
    config: dict
        Top-level YAML configuration dict for the bin prioritizers containing
        `named_configs` and `bin_configs` entires (see default JEWEL config
        file for an example)

    Returns
    -------
    prioritizer: callable
        a prioritization function constructed by calling the `prioritizer`
        function with the parameters parsed from the YAML configuration
    """
    named_configs = {
        k:load_bin_prioritizer(v)
        for k, v in config['named_configs'].items()
    }
    bin_configs = config.get('bin_configs', {})
    return prioritizer(bin_configs, named_configs)
