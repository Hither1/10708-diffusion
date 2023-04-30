import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import warnings

def get_image(locs, colors, im_shape=(50, 50)):
    scene_image = np.ones((*im_shape, 3), dtype=np.uint8)
    scene_image[0] = 128 # Adding a horizontal line between 2 samples
    scene_image[locs[:, 0], locs[:, 1]] = (colors * 255).astype(np.uint8)
    return scene_image

def get_image_mode(locs, colors, im_shape=(50, 50)):
    warnings.filterwarnings("ignore")
    fig = plt.figure(figsize=(im_shape[0] / 100, im_shape[1] / 100))
    plt.style.use('dark_background')
    plt.box(False)
    plt.xlim(0, im_shape[0] / 100)
    plt.ylim(0, im_shape[1] / 100)
   
    for i in range(locs.shape[0]):
        for j in range(locs.shape[1]):
            plt.scatter(locs[i, j, :, 1] / 100, locs[i, j, :, 0] / 100, marker='o', s=4, c=colors[i, j])
            plt.plot(locs[i, j, :, 1] / 100, locs[i, j, :, 0] / 100, color=colors[i, j])
        
    canvas = plt.get_current_fig_manager().canvas # get the canvas object
    canvas.draw() # update the plot

    width, height = canvas.get_width_height()
    scene_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close()
    
    return scene_image

def get_image_agent(locs, colors, im_shape=(50, 50)):
    warnings.filterwarnings("ignore")
    fig = plt.figure(figsize=(im_shape[0] / 100, im_shape[1] / 100))
    for i in range(0, locs.shape[0]):
        plt.style.use('dark_background')
        plt.box(False)
        xmin, xmax = 0, im_shape[0] / 100
        plt.xlim(xmin, xmax)
        plt.ylim(xmin, xmax)
        pos = locs[i] / 100
        filtered_data = [d for d in pos if xmin + 0.05 < d[..., 0] < xmax - 0.05 and xmin + 0.05 < d[..., 1] < xmax - 0.05]
        x, y = [d[..., 1] for d in filtered_data], [d[..., 0] for d in filtered_data]
        plt.scatter(x, y, marker='o', s=4)
        plt.plot(x, y)
        if len(x) > 0:
            plt.text(x[0], y[0], str(y[0]))
            plt.text(x[-1], y[-1], str(y[-1]))
        # plt.xticks([])
        # plt.yticks([])
        
    canvas = plt.get_current_fig_manager().canvas # get the canvas object
    canvas.draw() # update the plot

    width, height = canvas.get_width_height()
    scene_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close()
    
    return scene_image

def get_color(color_name):
    if color_name == 'black':
        color = np.array([0.0, 0.0, 0.0])
    elif color_name == 'white':
        color = np.array([1.0, 1.0, 1.0])
    elif color_name == 'red':
        color = np.array([1.0, 0.0, 0.0])
    elif color_name == 'green':
        color = np.array([0.0, 1.0, 0.0])
    elif color_name == 'blue':
        color = np.array([0.0, 0.0, 1.0])
    else:
        raise NotImplementedError
    return color


def get_mode_colors(num_modes, colormap='Greens'):
    # colors = matplotlib.colormaps[colormap]
    colors = matplotlib.cm.get_cmap(colormap)
    mode_colors = colors(np.linspace(0., 1., num_modes))
    return mode_colors


def get_range_color(values, colormap='winter'):
    colors = matplotlib.colormaps[colormap]
    range_colors = colors(values)
    return range_colors
