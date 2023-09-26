import amt_tools.tools as tools
import matplotlib.pyplot as plt


def plot_stacked_pitch_list_with_spectrogram(audio_features, stacked_pitch_list, hertz=False,
                                             point_size=5, include_axes=True,
                                             x_bounds=None, y_bounds=None,
                                             colors=None, labels=None):

    fig = plt.figure(tight_layout=True)
    hcqt = audio_features[0, :, :]
    plt.imshow(hcqt)
    plt.show()

    # # Loop through the stack of pitch lists, keeping track of the index
    # for idx, slc in enumerate(stacked_pitch_list.keys()):
    #     # Get the times and pitches from the slice
    #     times, pitch_list = stacked_pitch_list[slc]
    #     # Determine the color to use when plotting the slice
    #     color = 'k' if colors is None else colors[idx]
    #     # Determine the label to use when plotting the slice
    #     label = None if labels is None else labels[idx]
    #
    #     # Use the pitch_list plotting function
    #     fig = tools.plot_pitch_list(times=times,
    #                                 pitch_list=pitch_list,
    #                                 hertz=hertz,
    #                                 point_size=point_size,
    #                                 include_axes=include_axes,
    #                                 x_bounds=x_bounds,
    #                                 y_bounds=y_bounds,
    #                                 color=color,
    #                                 label=label,
    #                                 idx=idx,
    #                                 fig=fig)

    return fig
