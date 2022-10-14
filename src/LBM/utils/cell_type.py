from enum import Enum
import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np


class CellType(Enum):
    NOTHING = 0
    FLUID = 1
    OBSTACLE = 2
    EMPTY = 4
    INFLOW = 8
    OUTFLOW = 16
    OPEN = 32
    STICK = 64

    def __int__(self):
        return self.value

    @staticmethod
    def get_colormap():  # pragma: no cover
        """Compute a colormap to plot CellTypes with reasonable colors

        Returns:
            colormap, formatter, norm and norm bins
        """
        col_dict = {
            0: "black",
            1: "blue",
            2: "grey",
            4: "white",
            8: "green",
            16: "red",
            32: "yellow",
            64: "orange",
        }
        # We create a colormap from our list of colors
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        labels = np.array(
            [
                "NOTHING",
                "FLUID",
                "OBSTACLE",
                "EMPTY",
                "INFLOW",
                "OUTFLOW",
                "OPEN",
                "STICK",
            ]
        )
        len_lab = len(labels)
        ## Prepare bins for the normalizer
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

        norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        return cm, fmt, norm, norm_bins
