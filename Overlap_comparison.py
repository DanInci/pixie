import numpy as np
import os
import pandas as pd
import copy
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import scipy.stats as stats
from PIL import Image
import skimage.io as io


# first attempt to evaluate accuracy of different networks by comparing to gold standard contoured data

# read in TIFs containing ground truth contoured data, along with predicted segmentation
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/First_Run/'
image_direc = base_dir + 'Point23/'

deep_direc = base_dir + 'analyses/20190505_deepcell_old/'
plot_direc = deep_direc + 'figs/'

# files = ["interior_2", "interior_5", "interior_border_2", "interior_border_5",
#          "interior_border_border_2", "interior_border_border_5", "interior_border_border_20"]

files = ["interior_border_border_watershed_"]
        # "interior_30", "interior_border_30", "interior_border_border_30"

suffixs = ["epoch_20", "epoch_30", "epoch_40"]

# # for looking specifically at current batch
file_base = "mask_python_smoothed_interior_border_border_watershed_epoch_40"
file_base = "mask_interior_border_border_deepcell_old_epoch_30_7threshold_2cutoff"
file_suf = ""

gcloud_old = False

suffixs = [""]
for i in range(len(files)):
    file_base = files[i]
    for j in range(len(suffixs)):
        file_suf = suffixs[j]
        predicted_data = io.imread(deep_direc + '' + file_base + file_suf + '.tiff')
        predicted_data[predicted_data > 1] = 1

        if gcloud_old:
            contour_data = io.imread(image_direc + "Nuclear_Interior_Mask_padded.tif")
        else:
            contour_data = io.imread(image_direc + "Nuclear_Interior_Mask.tif")

        contour_data[contour_data > 1] = 2

        overlap = predicted_data + contour_data

        # generates labels (L) for each distinct object in the image, along with their indices
        # For some reason, the regionprops output is 0 indexed, such that the 1st cell appears at index 0.
        # However, the labeling of these cells is 1-indexed, so that the 1st cell is given the label of 1
        # It's hard to describe how dumb and confusing this is.
        # Cell 452, for example, can have its regionprops data output by accessing the 451st index

        predicted_L, predicted_idx = skimage.measure.label(predicted_data > 0, return_num=True, connectivity=1)
        predicted_props = skimage.measure.regionprops(predicted_L)

        contour_L, contour_idx = skimage.measure.label(contour_data, return_num=True, connectivity=1)
        contour_props = skimage.measure.regionprops(contour_L)

        #  determine how well the contoured data was recapitulated by the predicted segmentaiton data
        cell_frame = pd.DataFrame(columns=["contour_cell", "contour_cell_size", "predicted_cell", "predicted_cell_size",
                                           "percent_overlap", "merged", "split", "missing", "bad"], dtype="float")

        for contour_cell in range(1, contour_idx + 1):

            # generate a mask for the contoured cell, get all predicted cells that overlap the mask
            mask = contour_L == contour_cell
            overlap_id, overlap_count = np.unique(predicted_L[mask], return_counts=True)
            overlap_id, overlap_count = np.array(overlap_id), np.array(overlap_count)

            # remove cells that aren't at least 5% of current cell
            contour_cell_size = sum(sum(mask))
            idx = overlap_count > 0.05 * contour_cell_size
            overlap_id, overlap_count = overlap_id[idx], overlap_count[idx]

            # sort the overlap counts in decreasing order
            sort_idx = np.argsort(-overlap_count)
            overlap_id, overlap_count = overlap_id[sort_idx], overlap_count[sort_idx]

            # check and see if maps primarily to background
            if overlap_id[0] == 0:
                if overlap_count[0] / contour_cell_size > 0.8:
                    # more than 70% of cell is overlapping with background, classify it as missing
                    cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                                    "predicted_cell": 0, "predicted_cell_size": 0,
                                                    "percent_overlap": overlap_count / contour_cell_size, "merged": False,
                                                    "split": False, "missing": True, "bad": False}, ignore_index=True)
                    continue
                else:
                    # not missing, just bad segmentation
                    cell_frame = cell_frame.append(
                        {"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                         "predicted_cell": overlap_id[1], "predicted_cell_size": predicted_props[overlap_id[1] - 1].area,
                         "percent_overlap": overlap_count / contour_cell_size, "merged": False,
                         "split": False, "missing": False, "bad": True}, ignore_index=True)
                    continue
            else:
                # remove background as target cell and change cell size to for calculation
                if 0 in overlap_id:
                    keep_idx = overlap_id != 0
                    contour_cell_size -= overlap_count[~keep_idx][0]
                    overlap_id, overlap_count = overlap_id[keep_idx], overlap_count[keep_idx]

            # go through logic to determine relationship between overlapping cells
            if overlap_count[0] / contour_cell_size > 0.9:

                # if greater than 90% of pixels contained in first overlap, assign to that cell
                pred_cell = overlap_id[0]
                pred_cell_size = predicted_props[pred_cell - 1].area
                percnt = overlap_count[0] / contour_cell_size

                cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                                "predicted_cell": pred_cell, "predicted_cell_size": pred_cell_size,
                                                "percent_overlap": percnt, "merged": False, "split": False,
                                                "missing": False, "bad": False}, ignore_index=True)
            else:

                # Determine whether any other predicted cells contribute primarily to this contoured cell
                # Identify number of cells needed to get to 90% of total volume of cell
                # cum_size = overlap_count[0]
                # idx = 0
                #
                # # because we removed some of the cells, need to get to 90% of remaining volume of cell
                # max_fraction = np.sum(overlap_count) / contour_cell_size
                # # TODO iterate through all cells isntead
                # while (cum_size / contour_cell_size) < 0.9 * max_fraction:
                #     idx += 1
                #     cum_size += overlap_count[idx]
                #
                #     if idx > 20:
                #         raise Exception("Something failed in the while loop")

                # Figure out which of these cells have at least 80% of their volume contained in original cell
                split_flag = False
                # TODO check if first cell also has at least 80% of volume contained in contour cell?
                # TODO can keep a counter of number of cells that meet this criteria, if >2 then split?
                for cell in range(1, len(overlap_id)):
                    pred_cell_size = predicted_props[overlap_id[cell] - 1].area
                    percnt = overlap_count[cell] / contour_cell_size
                    if overlap_count[cell] / pred_cell_size > 0.7:
                        # multiple predicted cells were assigned to single target cell, hence split
                        split_flag = True
                        cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                                        "predicted_cell": overlap_id[cell], "predicted_cell_size": pred_cell_size,
                                                        "percent_overlap": percnt, "merged": False, "split": True,
                                                        "missing": False, "bad": False}, ignore_index=True)
                    else:
                        # this cell hasn't been split, just poorly assigned
                        cell_frame = cell_frame.append(
                            {"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                             "predicted_cell": overlap_id[cell], "predicted_cell_size": pred_cell_size,
                             "percent_overlap": percnt, "merged": False, "split": False,
                             "missing": False, "bad": True}, ignore_index=True)

                # assign the first cell, based on whether or not subsequent cells indicate split
                cell_frame = cell_frame.append({"contour_cell": contour_cell, "contour_cell_size": contour_cell_size,
                                                "predicted_cell": overlap_id[0], "predicted_cell_size": overlap_count[0],
                                                "percent_overlap": overlap_count[0] / contour_cell_size, "merged": False,
                                                "split": split_flag, "missing": False, "bad": False}, ignore_index=True)


        def outline_objects(L_matrix, list_of_lists):
            """takes in an L matrix generated by skimage.label, along with a list of lists, and returns a mask that has the
            pixels for all cells from each list represented as integer values for easy plotting"""

            L_plot = copy.deepcopy(L_matrix).astype(float)

            for idx, list in enumerate(list_of_lists):
                mask = np.isin(L_plot, list)

                # use a decimal so that it doesn't tag any of the actual cell IDs when reassigning
                L_plot[mask] = idx + 1.99

            L_plot[L_plot > idx + 2] = 1
            L_plot = np.around(L_plot)
            L_plot = L_plot.astype('int16')
            return L_plot


        # identify categories of poorly classified cells
        cell_frame = cell_frame[cell_frame["contour_cell_size"] > 10]
        cell_frame = cell_frame[cell_frame["predicted_cell_size"] > 10]
        missing_idx = cell_frame["missing"] == True

        # ignore those cells mapping to 0 (background), as these weren't actually merged into one cell
        merge_idx = np.logical_and(cell_frame["predicted_cell"].duplicated(), ~missing_idx)
        split_idx = cell_frame["split"] == 1

        # bad_idx = np.logical_or(np.logical_or(cell_frame["contour_cell_size"] / cell_frame["predicted_cell_size"] > 1.3,
        #                         cell_frame["contour_cell_size"] / cell_frame["predicted_cell_size"] < 0.7),
        #                         cell_frame["bad"] == 1)

        cell_frame.loc[merge_idx, "merged"] = True
        cell_frame.loc[missing_idx, "missing"] = True
        split_cells = cell_frame.loc[split_idx, "predicted_cell"]
        split_cells = [x for x in split_cells if x != 0]
        merged_cells = cell_frame.loc[merge_idx, "predicted_cell"]
        merged_cells = [x for x in merged_cells if x != 0]
        # bad_cells = cell_frame.loc[bad_idx, "predicted_cell"]
        bad_cells = cell_frame.loc[cell_frame["bad"] == 1, "predicted_cell"]
        bad_cells = [x for x in bad_cells if x != 0]
        bad_cells = [x for x in bad_cells if ~np.isin(x, split_cells)]
        merged_cells = [x for x in merged_cells if ~np.isin(x, bad_cells)]

        # TODO add missing and created cell categories
        def randomize_labels(label_map):
            """Takes in a labeled matrix and swaps the integers around so that color gradient has better contrast

            Inputs:
            label_map(2D numpy array): labeled TIF with each object assigned a unique value

            Outputs:
            swapped_map(2D numpy array): labeled TIF with object labels permuted"""

            max_val = np.max(label_map)
            for cell_target in range(1, max_val):
                swap_1 = cell_target
                swap_2 = np.random.randint(1, max_val)
                swap_1_mask = label_map == swap_1
                swap_2_mask = label_map == swap_2
                label_map[swap_1_mask] = swap_2
                label_map[swap_2_mask] = swap_1

            label_map = label_map.astype('int16')

            return label_map

        swapped = randomize_labels(copy.copy(contour_L))

        classify_outline = outline_objects(predicted_L, [split_cells, merged_cells, bad_cells])

        fig, ax = plt.subplots(nrows=1, ncols=2)

        cmap = mpl.colors.ListedColormap(['Black', 'Grey', 'Blue', 'Red', 'Yellow'])

        # set limits .5 outside true range
        mat = ax[0].imshow(classify_outline, cmap=cmap, vmin=np.min(classify_outline)-.5, vmax=np.max(classify_outline)+.5)
        ax[1].imshow(swapped)

        # tell the colorbar to tick at integers
        #cbar = fig.colorbar(mat, ticks=np.arange(np.min(classify_outline), np.max(classify_outline)+1))

        #cbar.ax.set_yticklabels(['Background', 'Normal', 'Split', 'Merged', 'Low Quality'])
        fig.tight_layout()

        fig.savefig(plot_direc + file_base + file_suf + '_color_map.tiff', dpi=200)

        io.imsave(plot_direc + file_base + file_suf + '_color_map_raw.tiff', classify_outline)
        io.imsave(plot_direc + file_base + file_suf + '_ground_truth_raw.tiff', swapped * 100)

        # make subplots for two simple plots
        fig, ax = plt.subplots(2, 1, figsize=(10,10))

        ax[0].scatter(cell_frame["contour_cell_size"], cell_frame["predicted_cell_size"])
        ax[0].set_xlabel("Contoured Cell")
        ax[0].set_ylabel("Predicted Cell")

        # compute percentage of different error types
        errors = np.array([len(set(merged_cells)), len(set(split_cells)), len(set(bad_cells))]) / len(set(cell_frame["predicted_cell"]))
        position = range(len(errors))
        categories = ["Merged", "Split", "Bad"]
        ax[1].bar(position, errors)

        ax[1].set_xticks(position)
        ax[1].set_xticklabels(categories)
        ax[1].set_title("Percentage of cells misclassified")

        fig.savefig(plot_direc + file_base + file_suf + '_stats.tiff', dpi=200)

        # prob_data = Image.open(deep_direc + file_base + '_border' + '.tiff')
        # prob_data = np.array(prob_data)
        # hist_data = prob_data.reshape(-1, 1).squeeze()
        # hist_data = [x for x in hist_data if x > 0.05]
        # fig, ax = plt.subplots()
        # ax.hist(hist_data, bins=np.arange(0,1.1, .1))
        # ax.set_ylim(0, 200000)
        # plt.xticks(np.arange(0, 1.1, 0.1))
        # fig.savefig(plot_direc + file_base + "_hist.tiff")

# combine all three plots above into one
# fig3 = plt.figure(figsize=(15,9))
# gs = mpl.gridspec.GridSpec(2,2)
# f3_ax1 = fig3.add_subplot(gs[0, 0])
# f3_ax1.scatter(cell_frame["contour_cell_size"], cell_frame["predicted_cell_size"])
# f3_ax1.set_xlabel("Contoured Cell")
# f3_ax1.set_ylabel("Predicted Cell")
#
#
# f3_ax2 = fig3.add_subplot(gs[1, 0])
# f3_ax2.set_title('gs[1, :-1]')
# f3_ax2.bar(position, errors)
# f3_ax2.set_xticks(position)
# f3_ax2.set_xticklabels(categories)
# f3_ax2.set_title("Percentage of cells misclassified")
#
# cmap = mpl.colors.ListedColormap(['Black', 'Grey', 'Blue', 'Red', 'Yellow'])
# f3_ax3 = fig3.add_subplot(gs[:, 1])
# f3_ax3.imshow(classify_outline, cmap=cmap, vmin=np.min(classify_outline)-.5, vmax=np.max(classify_outline)+.5)
#
# # set limits .5 outside true range
# mat = f3_ax3.imshow(classify_outline, cmap=cmap, vmin=np.min(classify_outline)-.5, vmax=np.max(classify_outline)+.5)
#
# # tell the colorbar to tick at integers
# cbar = fig3.colorbar(mat, ticks=np.arange(np.min(classify_outline), np.max(classify_outline)+1))
# cbar.ax.set_yticklabels(['Background', 'Accurate', 'Split', 'Merged', 'Low Accuracy'])

# fig3.savefig("Deep_Segmentation_Border.tif")


# histogram plotting
prob_maps = np.zeros((1024, 1024, 5))
prob_maps[:, :, 0] = io.imread(deep_direc + 'interior_10' + '_border' + '.tiff')
prob_maps[:, :, 1] = io.imread(deep_direc + 'interior_30' + '_border' + '.tiff')
prob_maps[:, :, 2] = io.imread(deep_direc + 'interior_border_10' + '_border' + '.tiff')
prob_maps[:, :, 3] = io.imread(deep_direc + 'interior_border_30' + '_border' + '.tiff')

original_values = prob_maps[:, :, 1].flatten()
delta_values = (prob_maps[:, :, 3] - prob_maps[:, :, 1]).flatten()
delta_values = prob_maps[:, :, 3].flatten()


heatmap, xedges, yedges = np.histogram2d(original_values, delta_values, bins=100, range=[[0, 1], [0, 1]])
np.quantile(heatmap, [.85, .90, .95, .98, .99])
heatmap[heatmap > 500] = 500
heatmap_log = np.log(heatmap + 0.1)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
xs = np.linspace(0, 1, 100)
xs2 = np.repeat(0, 100)
plt.imshow(heatmap.T, origin='lower', extent=extent)
plt.plot(xs, xs, '-r')
plt.savefig(deep_direc + '/figs/interior_border_30vsinterior30_hist.tiff')



# other code

# plot values for the different classes of cells

# create matrix to keep track of number of neighbors each cell has
contour_stats = pd.DataFrame(np.zeros((contour_idx + 1, 2)))
contour_stats.columns = ["neighbors_5", "neighbors_7"]

contour_L_5 = scipy.ndimage.grey_dilation(contour_L, (5, 5))
contour_L_7 = scipy.ndimage.grey_dilation(contour_L, (7, 7))

# count number of times each cell overlaps with neighbor
for cell in range(1,contour_idx):
    overlap_ids = np.unique(contour_L_7[contour_L == cell])
    if len(overlap_ids) > 1:
        # don't count overlap with self
        idx = overlap_ids != cell
        overlap_ids = overlap_ids[idx]
        for overlap in overlap_ids:
            contour_stats.iloc[overlap, 1] += 1


cell_frame['ratio'] = cell_frame['base_cell_size'] / cell_frame['mapped_cell_size']

# loop through different neighbor values, plotting each one
neighbor_num = [-1, 0, 1, 2]
neighbor_labels = ["All cells", "No Adjacent Cells", "One Adjacent Cell", "2+ Adjacent Cells"]
plot_scatter = False
plt.figure()

for idx, val in enumerate(neighbor_num):
    cell_frame_small = cell_frame.copy()
    cell_frame_small = cell_frame_small[cell_frame_small['base_cell_size'] > 1]
    cell_frame_small = cell_frame_small[cell_frame_small['mapped_cell_size'] != 0]

    if idx == 0:
        plot_ids = contour_stats.index

    else:
        plot_ids = contour_stats[contour_stats['neighbors_7'] == val].index

    cell_frame_small = cell_frame_small[cell_frame_small['base_cell'].isin(plot_ids)]

    plt.subplot(2, 4, idx + 5)
    if plot_scatter:
        plt.scatter(cell_frame_small['base_cell_size'], (cell_frame_small['mapped_cell_size']), s=10)
    else:
        heatmap, xedges, yedges = np.histogram2d(cell_frame_small['base_cell_size'].astype(float),
                                                 cell_frame_small['mapped_cell_size'].astype(float), bins=80)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')

    plt.title(neighbor_labels[idx])

    if idx == 0:
        plt.ylabel("Cell Size: DeepCell")

    if idx == 2:
        plt.xlabel("Cell Size: Contoured")




cell_frame_small = cell_frame_small[cell_frame_small['ratio'] < 1.5]

heatmap, xedges, yedges = np.histogram2d(cell_frame_small['base_cell_size'].astype(float), cell_frame_small['mapped_cell_size'].astype(float), bins=80)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()




seq1 = np.arange(0,10,1)
seq2 = np.arange(10,15,0.5)

example1 = pd.DataFrame(np.random.randn(10,3), columns=["One", "Two","Three"])




# identify cutoff values. Cells without issues are plotted in dark blue, small in light green, big in yellow
# if ratio of contoured / deep is greater than 1, that means the deep output is too small, and cells have been
# hyper segmented. If ratio is less than 1, that means deep ouput is too big, and cells have been merged together


# calculate percent of background pixels in true image that are included in cells in image 2
cell_frame['base_in_background'] = 0
cell_frame['mapped_in_background'] = 0

for idx, cell in enumerate(cell_frame['base_cell']):
    mask = contour_L == cell
    target = deep_L[mask]
    cell_frame.loc[idx, 'base_in_background'] = sum(target == 0) / np.sum(mask)


my_fig = plt.figure()
plt.hist(cell_frame['base_in_background'][cell_frame['base_cell'] > 1])
plt.xlabel("Percentage of contoured cell in background in deep map")
my_fig.savefig(plot_path + "Figure_4.pdf")


# this likely puts the value in the wrong row, but since using it for histogram only doesn't matter
for idx, cell in enumerate(cell_frame['mapped_cell'][cell_frame['mapped_cell'] > 0]):
     mask = deep_L == cell
     target = contour_L[mask]
     cell_frame.loc[idx, 'mapped_in_background'] = sum(target == 0) / np.sum(mask)


my_fig = plt.figure()
plt.hist(cell_frame['mapped_in_background'][cell_frame['base_cell'] > 1])
plt.xlabel("Percentage of deep cells in background in contoured map")
my_fig.savefig(plot_path + "Figure_5.pdf")


# This is likely an artifact due to imdilate command being run on deep images, which makes them much bigger