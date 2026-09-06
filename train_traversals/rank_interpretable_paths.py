import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import os.path as osp
import numpy as np
import pandas as pd
import json
from lib import create_summarizing_gif


########################################################################################################################
## The following attributes are available (as long as they have been used to traverse attribute space using           ##
## `traverse_attribute_space.py`):                                                                                    ##
##                                                                                                                    ##
##   'identity', 'face_width', 'face_height', 'race', 'gender', 'age', 'yaw', 'pitch', 'roll'                         ##
##   'au_1_Inner_Brow_Raiser', 'au_2_Outer_Brow_Raiser', 'au_4_Brow_Lowerer', 'au_5_Upper_Lid_Raiser',                ##
##   'au_6_Cheek_Raiser', 'au_9_Nose_Wrinkler', 'au_12_Lip_Corner_Puller', 'au_15_Lip_Corner_Depressor',              ##
##   'au_17_Chin_Raiser', 'au_20_Lip_stretcher', 'au_25_Lips_part', 'au_26_Jaw_Drop',                                 ##
##   'celeba_bangs', 'celeba_beard', 'celeba_eyeglasses', 'celeba_smiling', 'celeba_age'                              ##
##                                                                                                                    ##
## You may edit the `ATTRIBUTE_GROUPS` dictionary in order to form new attribute groups that can be used for ranking  ##
## the discovered paths.                                                                                              ##
########################################################################################################################

# Define attribute groups
ATTRIBUTE_GROUPS = {
    # DEV
    'DEV': ('au_12_Lip_Corner_Puller', 'identity'),
    # Face geometry
    'Face-W': ('face_width', 'identity'),
    'Face-H': ('face_height', 'identity'),
    'Face-WH': ('face_width', 'face_height', 'identity'),
    # Age (FairFace)
    'Age-FareFace': ('age', 'identity', 'gender', 'race'),
    # Age (CelebA/Talk-to-Edit)
    'Age-CelebA': ('celeba_age', 'identity', 'gender', 'race'),
    # Gender-Race
    'Gender': ('gender', 'race', 'age', 'celeba_age'),
    # Rotation (yaw, pitch, roll)
    'Rotation': ('yaw', 'pitch', 'roll', 'identity', 'age', 'celeba_age', 'race', 'gender', 'celeba_bangs',
                 'celeba_beard', 'celeba_eyeglasses', 'celeba_smiling'),
    # Smiling-AU12
    'Smiling-AU12': ('au_12_Lip_Corner_Puller', 'identity', 'gender', 'age', 'race'),
    # Smiling-CelebA
    'Smiling-CelebA': ('celeba_smiling', 'identity', 'gender', 'age', 'race'),
    # Brow Lowerer
    'Brow-Lowerer-AU4': ('au_4_Brow_Lowerer', 'identity', 'gender', 'age', 'race'),
    # Bangs
    'Bangs': ('celeba_bangs', 'identity')
}

# Set attributes min-max ranges
attribute_ranges = {
    # === Face detection (SFD) ===
    'face_width': np.array([0.0, 1.0]),
    'face_height': np.array([0.0, 1.0]),
    # === Identity comparison (ArcFace) ===
    'identity': np.array([0, 1.0]),
    # === Pose estimation (HopeNet) ===
    'yaw': np.array([-1.1, +1.1]),
    'pitch': np.array([-0.5, +0.5]),
    'roll': np.array([-0.3, +0.3]),
    # === FairFace ===
    'race': np.array([0.0, 1.0]),
    'age': np.array([0.0, 1.0]),
    'gender': np.array([0.0, 1.0]),
    # === AUs (DISFA) ===
    "au_1_Inner_Brow_Raiser": np.array([0.0, 5.0]),
    "au_2_Outer_Brow_Raiser": np.array([0.0, 5.0]),
    "au_4_Brow_Lowerer": np.array([0.0, 5.0]),
    "au_5_Upper_Lid_Raiser": np.array([0.0, 5.0]),
    "au_6_Cheek_Raiser": np.array([0.0, 5.0]),
    "au_9_Nose_Wrinkler": np.array([0.0, 5.0]),
    "au_12_Lip_Corner_Puller": np.array([0.0, 5.0]),
    "au_15_Lip_Corner_Depressor": np.array([0.0, 5.0]),
    "au_17_Chin_Raiser": np.array([0.0, 5.0]),
    "au_20_Lip_stretcher": np.array([0.0, 5.0]),
    "au_25_Lips_part": np.array([0.0, 5.0]),
    "au_26_Jaw_Drop": np.array([0.0, 5.0]),
    # === CelebA Attributes (Talk-to-Edit) ===
    'celeba_bangs': np.array([0.0, 1.0]),
    'celeba_beard': np.array([0.0, 1.0]),
    'celeba_eyeglasses': np.array([0.0, 1.0]),
    'celeba_smiling': np.array([0.0, 1.0]),
    'celeba_age': np.array([0.0, 1.0])
}


def l1(x):
    """Perform L1-normalization."""
    return x / np.abs(x).sum(axis=1, keepdims=True)


def traversal_indices(num_points, identity=False, two_sided=False):
    """Return ordered traversal indices, or midpoint distance for two-sided identity scores."""
    indices = np.arange(num_points)
    if identity and two_sided:
        return np.abs(indices - num_points // 2)
    return indices


def calculate_attribute_correlations(attributes_values, attributes, attr_ranges, two_sided=False):
    """Calculate every sample/path/attribute correlation in one vectorized operation.

    This is algebraically equivalent to ``cov(attribute, index) / std(index)`` used by
    the original nested implementation. ``attributes_values`` has shape [S, K, A, P].
    """
    num_points = attributes_values.shape[-1]
    if num_points < 2:
        raise ValueError("At least two traversal points are required to calculate correlations")

    lower = attr_ranges[:, 0].reshape(1, 1, -1, 1)
    span = (attr_ranges[:, 1] - attr_ranges[:, 0]).reshape(1, 1, -1, 1)
    scaled = np.clip(2.0 * (attributes_values - lower) / span - 1.0, -1.0, 1.0)

    indices = np.stack([
        traversal_indices(
            num_points,
            identity=(attribute == 'identity'),
            two_sided=two_sided,
        )
        for attribute in attributes
    ]).astype(scaled.dtype, copy=False)
    centered_indices = indices - indices.mean(axis=1, keepdims=True)
    centered_values = scaled - scaled.mean(axis=-1, keepdims=True)

    covariance_numerators = np.einsum(
        'skap,ap->ska', centered_values, centered_indices, optimize=True
    )
    denominators = np.sqrt(
        (num_points - 1) * np.square(centered_indices).sum(axis=1)
    )
    return covariance_numerators / denominators.reshape(1, 1, -1)


def _create_summarizing_gif_job(job):
    """Worker entry point for one independent summarizing GIF."""
    imgs_root, gif_filename, num_imgs, gif_size, gif_fps = job
    create_summarizing_gif(
        imgs_root=imgs_root,
        gif_filename=gif_filename,
        num_imgs=num_imgs,
        gif_size=gif_size,
        gif_fps=gif_fps,
    )
    return gif_filename


def save_results(attributes, attr_idx_corr, metric, interpretable_paths_dict, summary_md_dict, create_gifs=True,
                 top_k=3, num_imgs=7, gif_size=256, gif_fps=30, hashes_root=None, hashes=None,
                 interpretable_paths_root=None, gif_executor=None):
    """Save interpretable paths results and create summarizing GIFs.

    Args:
        attributes (list)               : list of attribute names
        attr_idx_corr (np.ndarray)      : attributes-to-path correlation matrix
        metric (str)                    : choose metric; correlation (corr) or l1-normalized correlation (corr_l1)
        interpretable_paths_dict (dict) : interpretable paths dictionary
        summary_md_dict (dict)          : summary md dictionary
        create_gifs (bool)              : create and save summarizing GIF files
        top_k (int)                     : create summarizing GIF files for the top-k interpretable paths for each
                                          attribute and for each latent code
        num_imgs (int)                  : number of images for the static sequence in GIF
        gif_size (int)                  : GIF height (its width will be (num_imgs + 1) * gif_size)
        gif_fps (int)                   : GIF frames per second
        hashes_root (str)               : latent codes hashes root directory
        hashes (list)                   : list of latent codes hashes
        interpretable_paths_root (str)  : set interpretable paths root directory
    """
    # Create output directory for given metric
    out_dir = osp.join(interpretable_paths_root, metric)
    os.makedirs(out_dir, exist_ok=True)

    # Build pandas data frame structure for attributes correlations
    attr_idx_corr_df = pd.DataFrame(attr_idx_corr)

    # Save attributes correlations df to csv file
    attr_idx_corr_df.to_csv(path_or_buf=osp.join(out_dir, 'attr_idx_{}.csv'.format(metric)),
                            header=attributes,
                            index_label="path_id",
                            float_format='%.3f')

    # Initialize dictionary of top-k interpretable paths (for each given attribute)
    top_k_paths = dict()
    for i in range(top_k):
        top_k_paths.update({i: []})

    # Get top-k paths and calculate the diagonal cross-attribute correlation matrix
    first_rows = []
    for t in range(attr_idx_corr_df.shape[1]):
        # Sort paths by t-th attribute (wrt to absolute correlation)
        attr_idx_corr_df_sorted_by_t = attr_idx_corr_df.sort_values(by=t, ascending=False)
        first_rows.append(attr_idx_corr_df_sorted_by_t.to_numpy()[0, :])

        # Get ranked interpretable paths
        interpretable_paths_dict[metric][attributes[t]] = attr_idx_corr_df_sorted_by_t.index.tolist()

        # Get ids of the top-k paths
        for i in range(top_k):
            top_k_paths[i].append(attr_idx_corr_df_sorted_by_t.index.tolist()[i])

        # Save attributes correlations df (sorted by attribute) to csv file
        attr_idx_corr_df_sorted_by_t.to_csv(
            path_or_buf=osp.join(out_dir, 'attr_idx_{}_sorted_by_{}.csv'.format(metric, attributes[t])),
            header=attributes,
            float_format='%.3f')

    # Save diagonal cross-attribute correlation matrix to csv file
    pd.DataFrame(np.stack(first_rows)).to_csv(path_or_buf=osp.join(out_dir, 'attr_idx_{}_diag.csv'.format(metric)),
                                              header=attributes,
                                              float_format='%.2f')

    # Create summarizing GIFs for each attribute and top-k paths
    if create_gifs:
        gif_jobs = []
        for a in range(len(attributes)):
            attr = attributes[a]
            attr_dir = osp.join(out_dir, attr)
            os.makedirs(attr_dir, exist_ok=True)
            for k in range(top_k):
                for h in hashes:
                    # Create summarizing GIF
                    imgs_root = osp.join(hashes_root, h, 'paths_images', 'path_{:03d}'.format(top_k_paths[k][a]))
                    gif_filename = osp.join(attr_dir, '{}_{}_{}_{}.gif'.format(attr, k + 1, top_k_paths[k][a], h))
                    gif_jobs.append((imgs_root, gif_filename, num_imgs, gif_size, gif_fps))
                    # Update interpretable paths dictionary
                    summary_md_dict[attr][h][metric][k+1] = top_k_paths[k][a]

        if gif_executor is None:
            for job in gif_jobs:
                _create_summarizing_gif_job(job)
        else:
            # Consume the iterator so worker errors are raised in the main process.
            for _ in gif_executor.map(_create_summarizing_gif_job, gif_jobs, chunksize=1):
                pass


def create_summary_md_file(attr_group, summary_md_dict, metric, top_k=3, hashes=None, interpretable_paths_root=None):
    """Create summary .md file for the given attributes group. For all given attributes, show the top-k interpretable
    paths for each latent code (hash).

    Args:
        attr_group (str)                : attributes group
        summary_md_dict (dict)          : summary md dictionary
        metric (str)                    : choose metric; correlation (corr) or l1-normalized correlation (corr_l1)
        top_k (int)                     : create summarizing GIF files for the top-k interpretable paths for each
                                          attribute and for each latent code
        hashes (list)                   : list of latent codes hashes
        interpretable_paths_root (str)  : set interpretable paths root directory

    """
    # Write .md summary filename
    md_summary_file = osp.join(interpretable_paths_root, 'top-{}_interpretable_path_{}.md'.format(top_k, attr_group))
    f = open(md_summary_file, "w")
    f.write("# Attribute group: {}\n".format(attr_group))
    attributes = tuple(a for a in ATTRIBUTE_GROUPS[attr_group] if a != 'identity')
    metrics = metric.split('+')
    for attr in attributes:
        f.write("## {}\n".format(attr))
        for h in hashes:
            f.write("### Latent code: {}\n".format(h))
            for m in metrics:
                f.write("#### Metric: {}\n".format(m))
                f.write("<p align=\"center\">\n")
                for k in range(top_k):
                    path_id = summary_md_dict[attr][h][m][k+1]
                    gif_file = osp.join(m, attr, "{}_{}_{}_{}.gif".format(attr, k+1, path_id, h))
                    gif_mouseover = "top-{} interpretable path [path_id: {}] for {}".format(k + 1, path_id, attr)
                    f.write("<img src=\"{}\" title=\"{}\"/>\n".format(gif_file, gif_mouseover))
                f.write("</p>\n")
    f.close()


def main():
    """A script for ranking the discovered non-linear paths in terms of correlation with a set of attributes.

    Options:
        ================================================================================================================
        -v, --verbose : set verbose mode on
        ================================================================================================================
        --exp         : set experiment's model dir, whose results/ directory contains sample traversal directories.
        --two-sided   : rank legacy negative-to-positive traversals using midpoint distance for identity. By default,
                        rankings assume one-sided traversals beginning at frame 0.
        --gif         : create summarizing GIF files
        --no-gif      : do NOT create summarizing GIF files
        --num-imgs    : set number of static images per sequence; by default all available images in the path will be
                        used
        --gif-size    : set GIF image size (otherwise use dimensions of given images)
        --gif-fps     : set number of frames per second for the generated GIF image
        --workers     : number of parallel GIF-generation workers
        --top-k       : create summarizing GIFs for the top-k interpretable path for each latent code and each attribute
        --attr-group  : choose one attribute group from `ATTRIBUTE_GROUPS`; by default rank every group
        --metric      : choose path ranking metric -- correlation (corr), l1-normalized correlation (corr_l1),
                        or both (corr+corr_l1)
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="WarpedGANSpace interpretable path ranking script")
    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    parser.add_argument('--exp', type=str, required=True,
                        help="set experiment's model dir (created by `train.py` and used by `traverse_latent_space.py` "
                             "and `traverse_attribute_space.py`.)")
    parser.add_argument('--two-sided', action='store_true',
                        help="use midpoint distance for identity scores from legacy two-sided traversals")
    parser.add_argument('--gif', dest='gif', action='store_true', help="create summarizing GIF files")
    parser.add_argument('--no-gif', dest='gif', action='store_false', help="do NOT create summarizing GIF files")
    parser.set_defaults(gif=True)
    parser.add_argument('--num-imgs', type=int, help="set number of static images per sequence; by default all "
                                                     "available images in the path will be used")
    parser.add_argument('--gif-size', type=int, default=256, help="GIF image size")
    parser.add_argument('--gif-fps', type=int, default=30, help="set GIF frame rate")
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1),
                        help="number of parallel GIF-generation workers (default: up to 8)")
    parser.add_argument('--top-k', type=int, default=3, help="create summarizing GIFs for the top-k interpretable path "
                                                             "for each latent code and each attribute")
    parser.add_argument('--attr-group', type=str, default=None, choices=ATTRIBUTE_GROUPS.keys(),
                        help="set one attribute group; by default rank every group in `ATTRIBUTE_GROUPS`")
    parser.add_argument('--metric', type=str, default='corr+corr_l1', choices=('corr', 'corr_l1', 'corr+corr_l1'),
                        help="choose path ranking metric -- correlation (corr), l1-normalized correlation (corr_l1) or "
                             "both (corr+corr_l1)")
    # Parse given arguments
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    # Check given experiment's directory
    latent_traversal_dir = osp.join(args.exp, 'results')
    if not osp.isdir(args.exp):
        raise NotADirectoryError("Error: invalid experiment's directory: {}".format(args.exp))
    else:
        if not osp.isdir(latent_traversal_dir):
            raise NotADirectoryError("Error: results directory not found: {}".format(latent_traversal_dir))

    sample_dirs = sorted(
        dirname for dirname in os.listdir(latent_traversal_dir)
        if osp.isdir(osp.join(latent_traversal_dir, dirname))
        and osp.isdir(osp.join(latent_traversal_dir, dirname, 'eval_np'))
        and osp.isdir(osp.join(latent_traversal_dir, dirname, 'paths_images'))
    )
    if not sample_dirs:
        raise FileNotFoundError("No evaluated sample traversal directories found under {}".format(latent_traversal_dir))

    selected_attr_groups = [args.attr_group] if args.attr_group is not None else list(ATTRIBUTE_GROUPS.keys())

    if args.verbose:
        print("#. Rank interpretable paths in {}".format(latent_traversal_dir))
        print("  \\__.Traversal mode: {}".format('two-sided' if args.two_sided else 'one-sided'))
        print("  \\__.Samples: {}".format(len(sample_dirs)))
        print("  \\__.Attribute groups: {}".format(selected_attr_groups))
        if args.gif:
            print("  \\__.GIF workers: {}".format(args.workers))

    # Reuse one pool across all groups and metrics to avoid repeated worker startup costs.
    gif_executor = ThreadPoolExecutor(max_workers=args.workers) if args.gif and args.workers > 1 else None

    # Calculate path-to-attribute correlations for every traversal path and selected attribute group.
    ranking_jobs = [
        (attr_group, latent_traversal_dir, sample_dirs)
        for attr_group in selected_attr_groups
    ]
    for attr_group, hashes_root, hashes in ranking_jobs:
        attributes = ATTRIBUTE_GROUPS[attr_group]
        attr_ranges = np.stack([attribute_ranges[attribute] for attribute in attributes])
        if args.verbose:
            print("       \\__.Attributes group '{}': {}".format(attr_group, attributes))

        # Get attributes for all samples and store them in a numpy array of shape:
        #   [num_samples, num_of_attributes, num_of_paths, num_of_path_steps]
        ATTRIBUTES = []
        for i in range(len(hashes)):
            if args.verbose:
                print("           \\__.hash: {} [{}/{}]".format(hashes[i], i + 1, len(hashes)))
            h_dir = osp.join(hashes_root, hashes[i])
            h_eval_np_dir = osp.join(h_dir, 'eval_np')
            SAMPLE_ATTRIBUTES = []
            for a in attributes:
                attr_traversal_file = osp.join(h_eval_np_dir, '{}.npy'.format(a))
                try:
                    SAMPLE_ATTRIBUTES.append(np.load(attr_traversal_file))
                except FileNotFoundError:
                    print("Attribute traversal file not found: {}".format(attr_traversal_file))
            ATTRIBUTES.append(SAMPLE_ATTRIBUTES)
        ATTRIBUTES = np.array(ATTRIBUTES)

        # Reshape as [num_samples, num_of_paths, num_of_attributes, num_of_path_steps]
        ATTRIBUTES = np.transpose(ATTRIBUTES, axes=(0, 2, 1, 3))

        # Create interpretable paths output directory
        interpretable_paths_root = osp.join(hashes_root, 'interpretable_paths', 'Group_{}'.format(attr_group))
        os.makedirs(interpretable_paths_root, exist_ok=True)

        # Save attributes traversals details
        num_of_samples = ATTRIBUTES.shape[0]
        num_of_paths = ATTRIBUTES.shape[1]
        num_of_attributes = ATTRIBUTES.shape[2]
        num_of_points_per_path = ATTRIBUTES.shape[3]
        if args.two_sided and num_of_points_per_path % 2 == 0:
            raise ValueError(
                "Two-sided rankings require an odd number of traversal points; found {}".format(
                    num_of_points_per_path
                )
            )
        if args.verbose:
            print("           \\__.Attributes matrix (ATTRIBUTES) : {}".format(ATTRIBUTES.shape))
            print("               \\__Number of samples           : {}".format(num_of_samples))
            print("               \\__Number of paths             : {}".format(num_of_paths))
            print("               \\__Number of attributes        : {}".format(num_of_attributes))
            print("               \\__Number of points per path   : {}".format(num_of_points_per_path))

        with open(osp.join(interpretable_paths_root, 'attributes_traversals_details.json'), 'w') as fp:
            json.dump({
                'traversal_mode': 'two-sided' if args.two_sided else 'one-sided',
                'num_of_samples': num_of_samples,
                'num_of_paths': num_of_paths,
                'num_of_attributes': num_of_attributes,
                'num_of_points_per_path': num_of_points_per_path
            }, fp)

        # Save attributes groups dictionary
        with open(osp.join(hashes_root, 'interpretable_paths', 'attributes_groups.json'), 'w') as fp:
            json.dump(ATTRIBUTE_GROUPS, fp)

        # Scale all attributes and calculate every sample/path correlation in vectorized NumPy kernels.
        ATTRIBUTES_IDX_CORR = calculate_attribute_correlations(
            ATTRIBUTES,
            attributes,
            attr_ranges,
            two_sided=args.two_sided,
        )

        # Average over samples
        ATTRIBUTES_IDX_CORR = ATTRIBUTES_IDX_CORR.mean(0)

        # Save results
        if args.verbose:
            print("           \\__.Save results...")

        # Initialize interpretable paths dictionary
        interpretable_paths_dict = dict()
        for m in ('corr', 'corr_l1'):
            m_dict = dict()
            for a in attributes:
                m_dict.update({a: []})
            interpretable_paths_dict.update({m: m_dict})

        # Initialize summary md dictionary
        summary_md_dict = dict()
        for a in attributes:
            a_dict = dict()
            for h in hashes:
                a_h_dict = dict()
                for m in ('corr', 'corr_l1'):
                    a_h_m_dict = dict()
                    for k in range(args.top_k):
                        a_h_m_dict.update({k+1: None})
                    a_h_dict.update({m: a_h_m_dict})
                a_h_dict.update({h: a_h_dict})
                a_dict.update({h: a_h_dict})
            summary_md_dict.update({a: a_dict})

        if args.metric in ('corr', 'corr+corr_l1'):
            # Save attribute correlation results
            if args.verbose:
                print("               \\__.Correlation...")

            save_results(attributes=list(attributes),
                         attr_idx_corr=np.abs(ATTRIBUTES_IDX_CORR),
                         metric='corr',
                         interpretable_paths_dict=interpretable_paths_dict,
                         summary_md_dict=summary_md_dict,
                         create_gifs=args.gif,
                         top_k=args.top_k,
                         num_imgs=args.num_imgs,
                         gif_size=args.gif_size,
                         gif_fps=args.gif_fps,
                         hashes_root=hashes_root,
                         hashes=hashes,
                         interpretable_paths_root=interpretable_paths_root,
                         gif_executor=gif_executor)

        if args.metric in ('corr_l1', 'corr+corr_l1'):
            # Save attribute l1-normalized correlation results
            if args.verbose:
                print("               \\__.Correlation (L1-normalized)...")

            save_results(attributes=list(attributes),
                         attr_idx_corr=l1(np.abs(ATTRIBUTES_IDX_CORR)),
                         metric='corr_l1',
                         interpretable_paths_dict=interpretable_paths_dict,
                         summary_md_dict=summary_md_dict,
                         create_gifs=args.gif,
                         top_k=args.top_k,
                         num_imgs=args.num_imgs,
                         gif_size=args.gif_size,
                         gif_fps=args.gif_fps,
                         hashes_root=hashes_root,
                         hashes=hashes,
                         interpretable_paths_root=interpretable_paths_root,
                         gif_executor=gif_executor)

        if args.verbose:
            print("           \\__.Create summary md file...")

        create_summary_md_file(attr_group=attr_group,
                               summary_md_dict=summary_md_dict,
                               metric=args.metric,
                               top_k=args.top_k,
                               hashes=hashes,
                               interpretable_paths_root=interpretable_paths_root)

        # Save interpretable paths dictionary
        with open(osp.join(interpretable_paths_root, 'interpretable_paths.json'), 'w') as fp:
            json.dump(interpretable_paths_dict, fp)

    if gif_executor is not None:
        gif_executor.shutdown()


if __name__ == '__main__':
    main()
