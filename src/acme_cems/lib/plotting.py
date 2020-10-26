# Functions for various plotting features of ACME analyzer
# these are called from analyzer.py
import os
import logging

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from PIL import Image

from acme_cems.lib.utils import find_nearest_index, make_crop

def get_axes_ticks_and_labels(mass_axis, time_axis):
    '''
        Sets heatmap axes values
    '''
    # extract zero padding if any
    padding_m = len(mass_axis[mass_axis == 0])//2
    padding_t = len(time_axis[time_axis == 0])//2

    time_ticks = []
    time_labels = []
    range_t = np.arange(0, int(time_axis[-padding_t-1]), 5)
    range_t[0] = 1
    for t in range_t:
        t_idx = find_nearest_index(time_axis, t)
        time_ticks.append(t_idx)
        time_labels.append(int(round(time_axis[t_idx], 2)))

    #TODO: check why we dont also compute mass_ticks in this fuction

    mass_labels = []
    nm = len(mass_axis) - padding_m *2 #count length of mass axis without the zero padding
    nm_labels = 8   # number of labels we like to have on heatmap axes
    step_m = max(int(nm / (nm_labels - 1)), 1)

    for m in range(50 + padding_m, nm, step_m):
        value = 5 * round(mass_axis[m] / 5)
        # make sure rounded value is not outside of mass axis range
        if (value <= np.max(mass_axis)) & (value >= np.min(mass_axis[mass_axis > 0])):
            mass_labels.append(int(value))  # round to the nearest multiple of 5

    return time_ticks, time_labels, mass_labels

def plot_heatmap(matrix, mass_axis, axes_info, title, label, file_id, outdir):
    '''
        Plots 2D heatmap
    '''
    time_ticks, time_labels, mass_labels = axes_info
    mass_ticks = []

    # flip image for plotting
    mass_axis = np.flip(mass_axis)
    matrix = np.flipud(matrix)

    for m in mass_labels:
        mass_ticks.append(find_nearest_index(mass_axis, m))

    exp_mean = np.mean(matrix)
    exp_std = np.std(matrix)
    matrix_clip = np.clip(matrix, 0, exp_mean + 3 * exp_std)

    plt.close('all')
    fig, ax = plt.subplots()
    cmap = ax.imshow(matrix_clip, cmap='inferno', interpolation=None, aspect='auto')
    cbar = fig.colorbar(cmap, ax=ax, label='Ion Counts (clipped at 3 std)')

    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)
    ax.set_yticks(mass_ticks)
    ax.set_yticklabels(mass_labels)

    plt.title(title)
    plt.suptitle(file_id)
    plt.xlabel('Time (Min)')
    plt.ylabel('Mass (AMU/Z)')
    plt.savefig(os.path.join(outdir, "Heat_Maps", file_id + label + '.png'), dpi=200)

def plot_heatmap_with_peaks(matrix, peaks_coord, mass_axis, axes_info, title, label, file_id, outdir, savedata):
    '''
        Plots 2D heatmaps with peaks
    '''
    time_ticks, time_labels, mass_labels = axes_info
    mass_ticks = []

    # flip image for plotting
    mass_axis = np.flip(mass_axis)
    matrix = np.flipud(matrix)
    peaks_coord['y'] = matrix.shape[0] - peaks_coord['y']

    for m in mass_labels:
        mass_ticks.append(find_nearest_index(mass_axis, m))

    exp_mean = np.mean(matrix)
    exp_std = np.std(matrix)
    matrix_clip = np.clip(matrix, 0, exp_mean + 3 * exp_std)

    plt.close('all')
    fig, ax = plt.subplots()

    ax.scatter(peaks_coord['x'], peaks_coord['y'], c='lightgreen', s=.2)
    cmap = ax.imshow(matrix_clip, aspect='auto', cmap='inferno')
    cbar = fig.colorbar(cmap, ax=ax, label='Ion Counts (clipped at 3 std)')

    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)
    ax.set_yticks(mass_ticks)
    ax.set_yticklabels(mass_labels)

    plt.title(title)
    plt.suptitle(file_id)
    plt.xlabel('Time (Min)')
    plt.ylabel('Mass (AMU/Z)')
    plt.savefig(os.path.join(outdir, "Heat_Maps", file_id + label + '.png'), dpi=200)

    if savedata:
        data = {'X': peaks_coord['x'],
                'Y': peaks_coord['y']}
        data_pd = pd.DataFrame.from_dict(data, orient='index').transpose()
        data_pd.to_csv(os.path.join(outdir, "Heat_Maps", file_id + label + '.csv'), index=False)

def plot_peak_vs_time(label, peak_properties, debug_plot, exp, mass_axis, time_axis, outdir, center_x, window_x, window_y, trace_window,knowntraces, compounds):

    logging.info(f'{label}: Plotting peaks vs Time.')
    if debug_plot:  # make a plot of every peak, centered at this peak
        window = 300
        for peak in peak_properties.itertuples():
            try:
                plot_name = 'Mass_' + str(round(mass_axis[peak.mass_idx], 2)) + '_' + str(
                    round(time_axis[peak.time_idx], 2))
                savepath = os.path.join(outdir, "Debug_Plots", plot_name + '.png')

                trace = exp[peak.mass_idx, :]

                background_range = [peak.time_idx - (center_x // 2),
                                    peak.time_idx + (center_x // 2) + 1,
                                    peak.time_idx - (window_x // 2),
                                    peak.time_idx + (window_x // 2) + 1]

                plt.close('all')
                plt.figure(figsize=(10, 5))

                # plot raw data slices in window_y
                for i in range(window_y):
                    trace_i = exp[peak.mass_idx + i - (window_y // 2), :]
                    plt.plot(time_axis, trace_i, '.-m', alpha=0.1, label='Raw Data off-center')

                # plot background range, offset and standard deviation
                plt.axhline(peak.background_abs, c='k', ls='--', alpha=0.3, label='Background offset')
                plt.axhline(peak.background_std, c='k', alpha=0.3, label='Background 1 std')

                for b in background_range:
                    plt.axvline(time_axis[b], alpha=0.3, c='k', label='Background Time Range')

                # plot calculated peak height
                plt.axhline(peak.height, c='c', alpha=0.3, label='Peak Height')
                plt.axhline(peak.height + peak.background_abs, c='y', alpha=0.3, label='Peak Height + Background offset')

                # plot raw data slice
                plt.plot(time_axis, trace, '.-r', alpha=0.5, label='Raw Data')

                # plot found peak, start and end time
                plt.plot(time_axis[peak.time_idx], trace[peak.time_idx], '*g', alpha=0.5, markersize=10, label='Peak')
                plt.plot(time_axis[peak.start_time_idx], trace[peak.start_time_idx], '.b', alpha=0.5, markersize=10, label='Start')
                plt.plot(time_axis[peak.end_time_idx], trace[peak.end_time_idx], '.b', alpha=0.5, markersize=10, label='End')

                plt.xlabel("Time (Min)")
                plt.ylabel("Ion Counts")
                plt.title('Raw Data with identified Peak +-150 time steps')
                x1 = np.max([0, peak.time_idx - window // 2])
                x2 = np.min([len(time_axis) - 1, peak.time_idx + window // 2])
                plt.xlim(time_axis[x1], time_axis[x2])
                plt.ylim(0, peak.height * 1.1 + peak.background_abs)

                # make sure that every label is only plotted once
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())

                plt.savefig(savepath, dpi=200)
            except:
                logging.error(f'{label}: Peak at edge error')

    # check whether there are multiple peaks for a given mass
    masses_bin_idx = []
    m_neighbors = 0
    masses = np.unique(peak_properties.mass_idx.to_numpy())  # find unique masses
    # TODO: There is probably a better method to find these mass bins
    for m in masses:
        if np.abs(m - m_neighbors) > trace_window / 2:  # check whether we already processed this mass
            # average masses that are less than half the binning window seperated
            m_neighbors = masses[np.abs(m - masses) < trace_window / 2]
            m_neighbors = np.mean(m_neighbors)
            masses_bin_idx.append(int(np.mean(m_neighbors)))

    # find peaks in proximity to masses_bin_idx list
    for bin_idx in masses_bin_idx:
        peaks = peak_properties[np.abs(peak_properties.mass_idx.to_numpy() - bin_idx) < trace_window / 2]

        if knowntraces:
            # add compound name
            compounds_amu = np.array(list(compounds.keys()))
            mass_dist = np.abs(mass_axis[bin_idx] - compounds_amu)  # find distance between peak mass and known compounds mass
            best_fit = np.argmin(mass_dist)  # find distance to closest known compound mass
            compound_name = compounds[compounds_amu[best_fit]]
            plot_name = compound_name
        else:
            plot_name = 'Mass_' + str(round(mass_axis[bin_idx], 2))

        savepath = os.path.join(outdir, "Time_Trace", plot_name + '.png')

        m1 = int(bin_idx - (trace_window // 2))
        m2 = int(bin_idx + trace_window // 2) + 1
        trace = np.sum(exp[m1:m2, :], 0)

        plt.close('all')
        plt.figure(figsize=(20, 5))
        plt.plot(time_axis, trace, 'r', alpha=0.8, label='Raw Data')
        plt.xlabel("Time (Min)")
        plt.ylabel("Ion Counts")
        plt.title('Raw Data for given Mass bin +-0.5 amu')

        # plot found peaks in one plot
        for peak in peaks.itertuples():
            plt.plot(time_axis[peak.time_idx], trace[peak.time_idx], '*g', alpha=0.5, markersize=10, label='Peak')
            plt.plot(time_axis[peak.start_time_idx], trace[peak.start_time_idx], '.b', alpha=0.5, markersize=10, label='Start')
            plt.plot(time_axis[peak.end_time_idx], trace[peak.end_time_idx], '.b', alpha=0.5, markersize=10, label='End')

        x1 = 0
        x2 = len(time_axis) - 1
        plt.xlim(time_axis[x1], time_axis[x2])

        # make sure that every label is only plotted once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.savefig(savepath, dpi=200)

def plot_peak_vs_mass(label, peak_properties, debug_plot, exp, mass_axis, time_axis, outdir, center_x, window_x, window_y, trace_window, exp_no_background):
    ''' plots peaks vs mass axis and saves them as png

    Parameters
    ----------
    peak_properties: DataFrame
        calculated properties of peaks (height, volumen, mass, time, ...)

    Returns
    -------
    plots as png

    '''
    logging.info(f'{label}: Plotting peaks vs Mass ...')
    if debug_plot:  # make a plot of every peak, centered at this peak
        window = 300
        for peak in peak_properties.itertuples():
            try:
                plot_name = 'Time_' + str(round(time_axis[peak.time_idx], 2)) + '_' + str(
                    round(mass_axis[peak.mass_idx], 2))
                savepath = os.path.join(outdir, "Debug_Plots", plot_name + '.png')

                trace = exp[:, peak.time_idx]

                background_range = [peak.mass_idx - (window_y // 2),
                                    peak.mass_idx + (window_y // 2) + 1]

                plt.close('all')
                plt.figure(figsize=(10, 5))

                # plot raw data slices in window_y
                for i in range(0, window_x, 5):
                    trace_i = exp_no_background[:, peak.time_idx + i - (window_x // 2)]
                    plt.plot(mass_axis, trace_i, '.-m', alpha=0.1, label='Data off-center')

                # plot background range, offset and standard deviation
                plt.axhline(peak.background_abs, c='k', ls='--', alpha=0.3, label='Background offset')
                plt.axhline(peak.background_std, c='k', alpha=0.3, label='Background 1 std')

                for b in background_range:
                    plt.axvline(mass_axis[b], alpha=0.3, c='k', label='Mass Range Considered')

                # plot calculated peak height
                plt.axhline(peak.height, c='c', alpha=0.3, label='Peak Height')
                plt.axhline(peak.height + peak.background_abs, c='y', alpha=0.3,
                            label='Peak Height + Background offset')

                # plot raw data slice
                plt.plot(mass_axis, trace, '.-r', alpha=0.5, label='Data')

                # plot found peak, start and end time
                plt.plot(mass_axis[peak.mass_idx], trace[peak.mass_idx], '*g', alpha=0.5, markersize=10,
                         label='Peak')

                plt.xlabel("Mass [amu])")
                plt.ylabel("Ion Counts")
                plt.title('Raw Data with identified Peak +-150 mass steps')
                x1 = np.max([0, peak.mass_idx - window // 2])
                x2 = np.min([len(mass_axis) - 1, peak.mass_idx + window // 2])
                plt.xlim(mass_axis[x1], mass_axis[x2])
                plt.ylim(0, peak.height * 1.1 + peak.background_abs)

                # make sure that every label is only plotted once
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())

                plt.savefig(savepath, dpi=200)
            except:
                logging.error(f'{label}: Peak at edge error')

    # check whether there are multiple peaks for a given mass
    times_bin_idx = []
    t_neighbors = 0
    times = np.unique(peak_properties.time_idx.to_numpy())  # find unique masses
    # TODO: There is probabily a better method to find these time bins

    for t in times:
        if np.abs(t - t_neighbors) > center_x:  # check whether we already processed this mass
            # average masses that are less than half the binning window seperated
            t_neighbors = times[np.abs(t - times) < center_x]
            t_neighbors = np.mean(t_neighbors)
            times_bin_idx.append(int(t_neighbors))

    # find peaks in proximity to time_bin_idx list
    for bin_idx in times_bin_idx:
        peaks = peak_properties[np.abs(peak_properties.time_idx.to_numpy() - bin_idx) < center_x]

        plot_name = 'Time_' + str(round(time_axis[bin_idx], 2))
        savepath = os.path.join(outdir, "Mass_Spectra", plot_name + '.png')

        t1 = int(bin_idx - (center_x))
        t2 = int(bin_idx + center_x) + 1
        trace = np.sum(exp[:, t1:t2], 1)
        trace_no_background = np.sum(exp_no_background[:, t1:t2], 1)

        plt.close('all')
        plt.figure(figsize=(20, 5))
        plt.plot(mass_axis, trace, 'r', alpha=0.3, label='Raw Data')
        plt.plot(mass_axis, trace_no_background, 'k', alpha=0.8, label='Data - Background')
        plt.xlabel("Mass [amu]")
        plt.ylabel("Ion Counts")
        plt.title('Raw Data for given Time bin +-10 sec')

        # plot found peaks in one plot
        for peak in peaks.itertuples():
            plt.plot(mass_axis[peak.mass_idx], trace_no_background[peak.mass_idx], '*g', alpha=0.5,
                     markersize=10, label='Peak')

        x1 = 0
        x2 = len(mass_axis) - 1
        plt.xlim(mass_axis[x1], mass_axis[x2])
        plt.ylim(0, np.max(trace_no_background) * 1.05)

        # make sure that every label is only plotted once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.savefig(savepath, dpi=200)

def plot_peak_vs_mass_time(label, peak_properties, exp_no_background, mass_axis, time_axis, outdir, center_x, window_x, window_y):
    ''' plots peaks vs mass and time and saves them as png

    Parameters
    ----------
    peak_properties: DataFrame
        calculated properties of peaks (height, volumen, mass, time, ...)

    Returns
    -------
    plots as png

    '''
    logging.info(f'{label}: Plotting peaks vs Time and Mass.')
    # make a plot of every peak, centered at this peak
    window_x_plot = window_x + 42  # extend in time we plot (make (window_x-1)%5==0 for nice plotting )
    window_y_plot = window_y + 40  # extend in mass we plot (make (window_x-1)%5==0 for nice plotting )

    for peak in peak_properties.itertuples():
        try:
            plot_name = 'TimeMass_' + str(round(time_axis[peak.time_idx], 2)) + '_' + str(
                round(mass_axis[peak.mass_idx], 2))
            savepath = os.path.join(outdir, "Debug_Plots", plot_name + '.png')

            # crop area centered on peak
            crop, _, _ = make_crop([peak.mass_idx, peak.time_idx], exp_no_background, window_x_plot + 2,
                                             window_y_plot, window_x_plot)
            exp_mean = np.mean(crop)
            exp_std = np.std(crop)
            matrix_clip = np.clip(crop, 0, exp_mean + 3 * exp_std)

            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 5))

            # prepare values for rectangle to show area considered as background
            # make right rectangle
            r_top_left = (window_x_plot // 2 + center_x / 2, window_y_plot // 2 - window_y / 2)
            r_height = window_y
            r_width = (window_x - center_x) / 2
            rectangle_r = plt.Rectangle(r_top_left, r_width, r_height, ec='green', fill=False, label='Background')
            # make left rectangle
            r_top_left = (window_x_plot // 2 - center_x / 2 - r_width, window_y_plot // 2 - window_y / 2)
            rectangle_l = plt.Rectangle(r_top_left, r_width, r_height, ec='green', fill=False, label='Background')
            # make rectangle to show center
            r_top_left = (window_x_plot // 2 - center_x / 2, window_y_plot // 2 - window_y / 2)
            r_width = center_x
            rectangle_c = plt.Rectangle(r_top_left, r_width, r_height, ec='red', ls='--', fill=False, label='Center')

            # plot rectangles
            fig.gca().add_patch(rectangle_r)
            fig.gca().add_patch(rectangle_l)
            fig.gca().add_patch(rectangle_c)

            # plot peak
            ax.plot(window_x_plot // 2, window_y_plot // 2, linestyle='none', markersize=5, marker="+", fillstyle='none',
                    c='r', label='Peak')
            # plot peak start/end time
            ax.axvline(peak.start_time_idx - peak.time_idx + (window_x_plot // 2), ls='--', label='Peak Start Time')
            ax.axvline(peak.end_time_idx - peak.time_idx + (window_x_plot // 2), ls='--', label='Peak End Time')
            cmap = ax.imshow(matrix_clip, cmap='inferno')
            fig.colorbar(cmap, ax=ax, label='Ion Counts (clipped at 3 std)')

            # make x/y legends
            time_labels = np.arange(0, matrix_clip.shape[1], 5)
            time_labels -= matrix_clip.shape[1] // 2
            mass_labels = np.arange(0, matrix_clip.shape[0], 5)
            mass_labels -= matrix_clip.shape[0] // 2
            ax.set_xticks(np.arange(0, matrix_clip.shape[1], 5))
            ax.set_xticklabels(time_labels)
            ax.set_yticks(np.arange(0, matrix_clip.shape[0], 5))
            ax.set_yticklabels(mass_labels)

            plt.ylabel("Mass [mass idx relative to peak]")
            plt.xlabel("Time [time idx relative to peak]")
            plt.title('Background removed Data with identified Peak')

            # handle background label being shown twice
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.savefig(savepath, dpi=200)
        except:
            logging.error(f'{label}: Peak at edge error')

def plot_mugshots(label, peak_properties, exp, time_axis, mass_axis, outdir):
    ''' plots mugshots of peaks and saves them as tif

    Parameters
    ----------
    peak_properties: DataFrame
        calculated properties of peaks (height, volumen, mass, time, ...)

    Returns
    -------
    plots as tif

    '''
    logging.info(f'{label}: Plotting mugshots of peaks.')
    # make a plot of every peak, centered at this peak
    # expand window to show some of the background if desired
    # Do not exceed 2x zero padding, which is currently set to window_x = 61
    # If these values are updated they must also be updated in background.py/overlay_peaks() !
    mug_x = 121 # max 121
    mug_y = 13  # max 25

    # iterate over peaks
    for peak in peak_properties.itertuples():
        # crop area centered on peak
        crop, _, _ = make_crop([peak.mass_idx, peak.time_idx], exp, mug_x + 2, mug_y, mug_x)

        # convert to values between 0 and 255
        max_val = np.max(crop)
        mugshot = crop / max_val * 255
        mugshot = mugshot.astype(np.uint8)

        # save image
        plot_name = f"Time_Mass_Max_{time_axis[peak.time_idx]:.2f}_{mass_axis[peak.mass_idx]:.2f}_{int(max_val)}"
        savepath = os.path.join(outdir, "Mugshots", plot_name + '.tif')
        im = Image.fromarray(mugshot, mode='L')
        im.save(savepath,'tiff')


        