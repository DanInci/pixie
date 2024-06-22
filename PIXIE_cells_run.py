import argparse

from ark.phenotyping import cell_cluster_utils, weighted_channel_comp, cell_som_clustering, cell_meta_clustering
import os
import json
import feather
import pandas as pd

from ark.utils.metacluster_remap_gui import metaclusterdata_from_files

SEED = 42


def merge_with_results(cell_table, cluster_counts_size_norm, appendix=''):
    results = cluster_counts_size_norm.rename(
        {'segmentation_label': 'label'}, axis=1
    )

    # merge the cell table with the consensus data to retrieve the meta clusters
    cell_table_merged = cell_table.merge(
        results, how='left', on=['fov', 'label']
    )

    # rename merged table results columns
    cell_table_merged = cell_table_merged.rename({
        'cell_som_cluster': f'pred_som_cluster{appendix}',
        'cell_meta_cluster_rename': f'pred_meta_cluster{appendix}'
    }, axis=1)

    return cell_table_merged


def main(config_path):
    # load the params
    with open(config_path) as f:
        pixie_config = json.load(f)

    # assign the params to variables
    input_dir = pixie_config['input_dir']
    output_dir = pixie_config['output_dir']
    fovs = pixie_config['fovs']

    channels = pixie_config['channels']

    print(f'Data Folder: {input_dir}\n')
    print(f'Output Folder: {output_dir}\n')
    print(f'FOVS: {fovs}\n')
    print(f'Channels to use: {channels}\n')

    # define the output directory of the pixel clustering
    pixel_output_dir = 'pixel_output'

    # define the base output cell folder
    cell_output_dir = 'cell_output'
    if not os.path.exists(os.path.join(output_dir, cell_output_dir)):
        os.mkdir(os.path.join(output_dir, cell_output_dir))

    # define the name of the directory with the extracted image data
    tiff_dir = os.path.join(input_dir, "images")

    # define the name of the directory with segmentation masks
    masks_dir = os.path.join(input_dir, "masks")

    # define the cell table path
    cell_table_path = os.path.join(input_dir, 'cells.csv')

    # define suffix of the segmentation mask files
    seg_suffix = '_whole_cell.tiff'

    # define the name of the cell clustering params file
    cell_clustering_params_name = 'cell_clustering_params.json'

    # load the pixel clustering params
    with open(os.path.join(output_dir, pixel_output_dir, cell_clustering_params_name)) as fh:
        cell_clustering_params = json.load(fh)

    # assign the params to variables
    pixel_data_dir = cell_clustering_params['pixel_data_dir']
    pc_chan_avg_som_cluster_name = cell_clustering_params['pc_chan_avg_som_cluster_name']
    pc_chan_avg_meta_cluster_name = cell_clustering_params['pc_chan_avg_meta_cluster_name']

    def get_method_config(pixel_cluster_col):
        config = {}

        config['pixel_cluster_col'] = pixel_cluster_col

        # depending on which pixel_cluster_col is selected, choose the pixel channel average table accordingly
        if pixel_cluster_col == 'pixel_som_cluster':
            config['pc_chan_avg_name'] = pc_chan_avg_som_cluster_name
            config['method_dir'] = 'from_som'
        elif pixel_cluster_col == 'pixel_meta_cluster_rename':
            config['pc_chan_avg_name'] = pc_chan_avg_meta_cluster_name
            config['method_dir'] = 'from_meta'

        method_dir = config['method_dir']
        if not os.path.exists(os.path.join(output_dir, cell_output_dir, method_dir)):
            os.mkdir(os.path.join(output_dir, cell_output_dir, method_dir))

        config["cell_som_weights_name"] = os.path.join(cell_output_dir, method_dir, 'cell_som_weights.feather')
        config["cluster_counts_name"] = os.path.join(cell_output_dir, method_dir, 'cluster_counts.feather')
        config["cluster_counts_size_norm_name"] = os.path.join(cell_output_dir, method_dir,
                                                               'cluster_counts_size_norm.feather')
        config["weighted_cell_channel_name"] = os.path.join(cell_output_dir, method_dir,
                                                            'weighted_cell_channel.feather')
        config["cell_som_cluster_count_avg_name"] = os.path.join(cell_output_dir, method_dir,
                                                                 'cell_som_cluster_count_avg.csv')
        config["cell_meta_cluster_count_avg_name"] = os.path.join(cell_output_dir, method_dir,
                                                                  'cell_meta_cluster_count_avg.csv')
        config["cell_som_cluster_channel_avg_name"] = os.path.join(cell_output_dir, method_dir,
                                                                   'cell_som_cluster_channel_avg.csv')
        config["cell_meta_cluster_channel_avg_name"] = os.path.join(cell_output_dir, method_dir,
                                                                    'cell_meta_cluster_channel_avg.csv')
        config["cell_meta_cluster_remap_name"] = os.path.join(cell_output_dir, method_dir,
                                                              'cell_meta_cluster_mapping.csv')

        return config

    config = get_method_config('pixel_meta_cluster_rename')
    # generate the preprocessed data before
    cluster_counts, cluster_counts_size_norm = cell_cluster_utils.create_c2pc_data(
        fovs, os.path.join(output_dir, pixel_data_dir), cell_table_path, config['pixel_cluster_col']
    )

    config['cluster_counts'] = cluster_counts
    config['cluster_counts_size_norm'] = cluster_counts_size_norm

    # define the count columns found in cluster_counts_norm
    cell_som_cluster_cols = cluster_counts_size_norm.filter(
        regex=f'{config["pixel_cluster_col"]}.*'
    ).columns.values

    config['cell_som_cluster_cols'] = cell_som_cluster_cols

    # write the unnormalized input data to cluster_counts_name for reference
    feather.write_dataframe(
        cluster_counts,
        os.path.join(output_dir, config['cluster_counts_name']),
        compression='uncompressed'
    )

    # generate the weighted cell channel expression data
    pixel_channel_avg = pd.read_csv(os.path.join(output_dir, config['pc_chan_avg_name']))
    weighted_cell_channel = weighted_channel_comp.compute_p2c_weighted_channel_avg(
        pixel_channel_avg,
        channels,
        config['cluster_counts'],
        fovs=fovs,
        pixel_cluster_col=config['pixel_cluster_col']
    )

    config['weighted_cell_channel'] = weighted_cell_channel

    # write the data to weighted_cell_channel_name
    feather.write_dataframe(
        weighted_cell_channel,
        os.path.join(output_dir, config['weighted_cell_channel_name']),
        compression='uncompressed'
    )

    cell_pysom = cell_som_clustering.train_cell_som(
        fovs,
        output_dir,
        cell_table_path=cell_table_path,
        cell_som_cluster_cols=config['cell_som_cluster_cols'],
        cell_som_input_data=config['cluster_counts_size_norm'],
        som_weights_name=config['cell_som_weights_name'],
        num_passes=1,
        seed=SEED
    )

    config['cell_pysom'] = cell_pysom

    # use cell SOM weights to assign cell clusters
    cluster_counts_size_norm = cell_som_clustering.cluster_cells(
        output_dir,
        config['cell_pysom'],
        cell_som_cluster_cols=config['cell_som_cluster_cols']
    )

    config['cluster_counts_size_norm'] = cluster_counts_size_norm

    # generate the SOM cluster summary files
    cell_som_clustering.generate_som_avg_files(
        output_dir,
        config['cluster_counts_size_norm'],
        cell_som_cluster_cols=config['cell_som_cluster_cols'],
        cell_som_expr_col_avg_name=config['cell_som_cluster_count_avg_name']
    )

    max_k = pixie_config['cells']['meta_max_k']
    cap = pixie_config['cells']['meta_cap']
    print(f'For metaclustering using max_k: {max_k} and z-score cap: [-{cap}, +{cap}].\n')

    # run hierarchical clustering using average count of pixel clusters per cell SOM cluster
    cell_cc, cluster_counts_size_norm = cell_meta_clustering.cell_consensus_cluster(
        output_dir,
        cell_som_cluster_cols=config['cell_som_cluster_cols'],
        cell_som_input_data=config['cluster_counts_size_norm'],
        cell_som_expr_col_avg_name=config['cell_som_cluster_count_avg_name'],
        max_k=max_k,
        cap=cap,
        seed=SEED,
    )

    config['cell_cc'] = cell_cc
    config['cluster_counts_size_norm'] = cluster_counts_size_norm

    # generate the meta cluster summary files
    cell_meta_clustering.generate_meta_avg_files(
        output_dir,
        config['cell_cc'],
        cell_som_cluster_cols=config['cell_som_cluster_cols'],
        cell_som_input_data=config['cluster_counts_size_norm'],
        cell_som_expr_col_avg_name=config['cell_som_cluster_count_avg_name'],
        cell_meta_expr_col_avg_name=config['cell_meta_cluster_count_avg_name']
    )

    # generate weighted channel summary files
    weighted_channel_comp.generate_wc_avg_files(
        fovs,
        channels,
        output_dir,
        config['cell_cc'],
        cell_som_input_data=config['cluster_counts_size_norm'],
        weighted_cell_channel_name=config['weighted_cell_channel_name'],
        cell_som_cluster_channel_avg_name=config['cell_som_cluster_channel_avg_name'],
        cell_meta_cluster_channel_avg_name=config['cell_meta_cluster_channel_avg_name']
    )

    cell_mcd = metaclusterdata_from_files(
        os.path.join(output_dir, config['cell_som_cluster_count_avg_name']),
        cluster_type='cell',
        prefix_trim=config['pixel_cluster_col'] + '_'
    )
    cell_mcd.output_mapping_filename = os.path.join(output_dir, config['cell_meta_cluster_remap_name'])
    cell_mcd.save_output_mapping()

    # rename the meta cluster values in the cell dataset
    cluster_counts_size_norm = cell_meta_clustering.apply_cell_meta_cluster_remapping(
        output_dir,
        config['cluster_counts_size_norm'],
        config['cell_meta_cluster_remap_name']
    )

    config['cluster_counts_size_norm'] = cluster_counts_size_norm

    # recompute the mean column expression per meta cluster and apply these new names to the SOM cluster average data
    cell_meta_clustering.generate_remap_avg_count_files(
        output_dir,
        config['cluster_counts_size_norm'],
        config['cell_meta_cluster_remap_name'],
        config['cell_som_cluster_cols'],
        config['cell_som_cluster_count_avg_name'],
        config['cell_meta_cluster_count_avg_name'],
    )

    # recompute the mean weighted channel expression per meta cluster and apply these new names to the SOM channel average data
    weighted_channel_comp.generate_remap_avg_wc_files(
        fovs,
        channels,
        output_dir,
        config['cluster_counts_size_norm'],
        config['cell_meta_cluster_remap_name'],
        config['weighted_cell_channel_name'],
        config['cell_som_cluster_channel_avg_name'],
        config['cell_meta_cluster_channel_avg_name']
    )

    cell_table = pd.read_csv(cell_table_path, dtype={
        'fov': str,
    })

    # merge results from PIXEL SOM clusters
    # cell_table = merge_with_results(cell_table, from_som_config['cluster_counts_size_norm'], '_from_pixel_som')

    # merge results from PIXEL META clusters
    cell_table = merge_with_results(cell_table, config['cluster_counts_size_norm'], '_from_pixel_meta')

    # subset on just the cell table columns plus the meta cluster rename column
    cell_table = cell_table[[
        'fov', 'label', 'cell_type',
        'pred_som_cluster_from_pixel_meta', 'pred_meta_cluster_from_pixel_meta'
    ]]

    # rename merged table columns for simplicity
    cell_table = cell_table.rename({
        'fov': 'sample_id',
        'label': 'object_id',
        'cell_type': 'label',
        'pred_som_cluster_from_pixel_meta': 'pred_som_cluster',
        'pred_meta_cluster_from_pixel_meta': 'pred_meta_cluster'
    }, axis=1)

    cell_table.to_csv(os.path.join(output_dir, 'pixie_results.csv'), index=False)

    feather.write_dataframe(
        cluster_counts_size_norm,
        os.path.join(output_dir, config['cluster_counts_size_norm_name']),
        compression='uncompressed'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description')

    # Define arguments
    parser.add_argument('--dataset', type=str, help='Dataset name', default='IMMUcan_2022_CancerExample')
    parser.add_argument('--blur_factor', type=int, help='Blur factor', default=2)
    parser.add_argument('--base_dir', type=str, help='Base directory', default='/home/dani/Documents/Thesis/Methods/IMCBenchmark')

    # Parse arguments
    args = parser.parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
    blur_factor = args.blur_factor

    config_path = os.path.join(base_dir, f'output/{dataset}/pixie/Blur={blur_factor}/config.json')
    main(config_path)
