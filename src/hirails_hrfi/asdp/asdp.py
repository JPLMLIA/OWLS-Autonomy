import os
import json
import numpy              as np
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

from skimage.color        import label2rgb
from skimage              import color
from PIL                  import Image

from utils.file_manipulation import tiff_read

def mugshots(hrfi_filepath, experiment, config):

    padding = config["mugshot_padding"]
    rgb = tiff_read(hrfi_filepath)
    gray = color.rgb2gray(rgb)

    image = np.zeros((rgb.shape[0],rgb.shape[1]))
    image[gray > 0.1] = 1

    asdp_dir = os.path.join(experiment, config['experiment_dirs']['asdp_dir'])
    mugshot_dir = os.path.join(asdp_dir, "mugshots")
    if not os.path.exists(mugshot_dir):
        os.makedirs(mugshot_dir)

    track_folder = os.path.join(experiment, config['experiment_dirs']['track_dir'])
    file_tag = hrfi_filepath.split("/")[-1].split(".")[0]
    with open(os.path.join(track_folder, f"{file_tag}_bboxes.json"), 'r') as f:
        label_image = json.load(f)

    if config["debug"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(rgb)

    mugshot_count = 0
    for region in label_image:
        minr = region[0] - padding
        minc = region[1] - padding
        maxr = region[2] + padding
        maxc = region[3] + padding
        mugshot = rgb[minr:maxr,minc:maxc,:]
        im = Image.fromarray(mugshot)
        im.save(os.path.join(mugshot_dir, f"{mugshot_count}.jpeg"))
        mugshot_count += 1

        if config["debug"]:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

    if config["debug"]:
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(asdp_dir,f"{file_tag}_mugshot_context.png"))
    
def generate_SUEs():
    print("HiRAILS SUEs")

def generate_DDs():
    print("HiRAILS DDs")