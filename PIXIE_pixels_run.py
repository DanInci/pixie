import argparse

from ark.phenotyping import pixie_preprocessing, pixel_som_clustering, pixel_meta_clustering
import os
import json

from ark.utils.metacluster_remap_gui import metaclusterdata_from_files

SEED = 42

def main(config_path):
    with open(config_path) as f:
        pixie_config = json.load(f)

    # assign the params to variables
    input_dir = pixie_config['input_dir']
    output_dir = pixie_config['output_dir']
    fovs = pixie_config['fovs']
    channels = pixie_config['channels']

    pixel_output_dir = 'pixel_output'

    if not os.path.exists(os.path.join(output_dir, pixel_output_dir)):
        os.makedirs(os.path.join(output_dir, pixel_output_dir))

    pixel_som_weights_name = os.path.join(pixel_output_dir, 'pixel_som_weights.feather')
    pc_chan_avg_som_cluster_name = os.path.join(pixel_output_dir, 'pixel_channel_avg_som_cluster.csv')
    pc_chan_avg_meta_cluster_name = os.path.join(pixel_output_dir, 'pixel_channel_avg_meta_cluster.csv')
    pixel_meta_cluster_remap_name = os.path.join(pixel_output_dir, 'pixel_meta_cluster_mapping.csv')

    # set to True to turn on multiprocessing
    multiprocess = pixie_config['pixels']['multiprocess']

    # define the number of FOVs to process in parallel, ignored if multiprocessing is set to False
    batch_size = pixie_config['pixels']['batch_size']

    print(f'Multiprocess: {multiprocess}')
    print(f'Batch size: {batch_size}')

    pixel_data_dir = os.path.join(pixel_output_dir, 'pixel_mat_data')
    pixel_subset_dir = os.path.join(pixel_output_dir, 'pixel_mat_subset')
    norm_vals_name = os.path.join(pixel_output_dir, 'channel_norm_post_rowsum.feather')

    channels = pixie_config['channels']
    type_channels = pixie_config['type_channels']

    blur_factor = pixie_config['pixels']['blur_factor']
    subset_proportion = pixie_config['pixels']['subset_proportion']

    print(f'Channels to use: {channels}\n')
    print(f'Blur Factor: {blur_factor}')
    print(f'Subset Proportion: {subset_proportion}\n')

    # define the name of the directory with the extracted image data
    tiff_dir = os.path.join(input_dir, "images")

    # define the name of the directory with segmentation masks
    masks_dir = os.path.join(input_dir, "masks")

    # define suffix of the segmentation mask files
    seg_suffix = '_whole_cell.tiff'

    pixie_preprocessing.create_pixel_matrix(
        fovs,
        channels,
        output_dir,
        tiff_dir,
        masks_dir,
        img_sub_folder=None,
        seg_suffix=seg_suffix,
        pixel_output_dir=pixel_output_dir,
        data_dir=pixel_data_dir,
        subset_dir=pixel_subset_dir,
        norm_vals_name=norm_vals_name,
        blur_factor=blur_factor,
        subset_proportion=subset_proportion,
        multiprocess=multiprocess,
        batch_size=batch_size,
        seed=SEED,
    )

    pixel_pysom = pixel_som_clustering.train_pixel_som(
        fovs,
        channels,
        output_dir,
        subset_dir=pixel_subset_dir,
        norm_vals_name=norm_vals_name,
        som_weights_name=pixel_som_weights_name,
        num_passes=1,
        seed=SEED
    )

    # use pixel SOM weights to assign pixel clusters
    pixel_som_clustering.cluster_pixels(
        fovs,
        channels,
        output_dir,
        pixel_pysom,
        data_dir=pixel_data_dir,
        multiprocess=multiprocess,
        batch_size=batch_size,
    )

    # generate the SOM cluster summary files
    pixel_som_clustering.generate_som_avg_files(
        fovs,
        channels,
        output_dir,
        pixel_pysom,
        data_dir=pixel_data_dir,
        pc_chan_avg_som_cluster_name=pc_chan_avg_som_cluster_name,
        seed=SEED,
    )

    max_k = pixie_config['pixels']['meta_max_k']
    cap = pixie_config['pixels']['meta_cap']
    print(f'For metaclustering using max_k: {max_k} and z-score cap: [-{cap}, +{cap}].\n')

    # run hierarchical clustering using average pixel SOM cluster expression
    pixel_cc = pixel_meta_clustering.pixel_consensus_cluster(
        fovs,
        channels,
        output_dir,
        max_k=max_k,
        cap=cap,
        data_dir=pixel_data_dir,
        pc_chan_avg_som_cluster_name=pc_chan_avg_som_cluster_name,
        multiprocess=multiprocess,
        batch_size=batch_size,
        seed=SEED,
    )

    # generate the meta cluster summary files
    pixel_meta_clustering.generate_meta_avg_files(
        fovs,
        channels,
        output_dir,
        pixel_cc,
        data_dir=pixel_data_dir,
        pc_chan_avg_som_cluster_name=pc_chan_avg_som_cluster_name,
        pc_chan_avg_meta_cluster_name=pc_chan_avg_meta_cluster_name,
        seed=SEED,
    )

    pixel_mcd = metaclusterdata_from_files(
        os.path.join(output_dir, pc_chan_avg_som_cluster_name),
        cluster_type='pixel',
        prefix_trim=None,
        subset_channels=type_channels
    )
    pixel_mcd.output_mapping_filename = os.path.join(output_dir, pixel_meta_cluster_remap_name)
    pixel_mcd.save_output_mapping()

    # rename the meta cluster values in the pixel dataset
    pixel_meta_clustering.apply_pixel_meta_cluster_remapping(
        fovs,
        channels,
        output_dir,
        pixel_data_dir,
        pixel_meta_cluster_remap_name,
        multiprocess=multiprocess,
        batch_size=batch_size
    )

    # recompute the mean channel expression per meta cluster and apply these new names to the SOM cluster average data
    pixel_meta_clustering.generate_remap_avg_files(
        fovs,
        channels,
        output_dir,
        pixel_data_dir,
        pixel_meta_cluster_remap_name,
        pc_chan_avg_som_cluster_name,
        pc_chan_avg_meta_cluster_name,
        seed=SEED,
    )

    # define the params dict
    cell_clustering_params = {
        'pixel_data_dir': pixel_data_dir,
        'pc_chan_avg_som_cluster_name': pc_chan_avg_som_cluster_name,
        'pc_chan_avg_meta_cluster_name': pc_chan_avg_meta_cluster_name
    }

    # save the params dict
    with open(os.path.join(output_dir, pixel_output_dir, 'cell_clustering_params.json'), 'w') as fh:
        json.dump(cell_clustering_params, fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description')

    # Define arguments
    parser.add_argument('--dataset', type=str, help='Dataset name', default='IMMUcan_2022_CancerExample')
    parser.add_argument('--blur_factor', type=str, help='Blur factor', default='Blur=2')
    parser.add_argument('--base_dir', type=str, help='Base directory', default='/home/dani/Documents/Thesis/Methods/IMCBenchmark')

    # Parse arguments
    args = parser.parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
    blur_factor = args.blur_factor

    config_path = os.path.join(base_dir, f'output/{dataset}/pixie/Blur={blur_factor}/config.json')
    main(config_path)
