import numpy as np


class ColorConfig:
    """
    Stores RGB values for each predicted output
    You can change colors to make it more attractive

    predicted output labels:
    1, 2 = Torso,
    3 = Right Hand,
    4 = Left Hand,
    5 = Left Foot,
    6 = Right Foot,
    7, 9 = Upper Leg Right,
    8, 10 = Upper Leg Left,
    11, 13 = Lower Leg Right,
    12, 14 = Lower Leg Left,
    15, 17 = Upper Arm Left,
    16, 18 = Upper Arm Right,
    19, 21 = Lower Arm Left,
    20, 22 = Lower Arm Right,
    23, 24 = Head;
    """
    COLORS = np.array([
        [20, 110, 255],
        [0, 150, 190],
        [133, 140, 197],
        [133, 140, 197],
        [204, 204, 0],
        [255, 255, 0],
        [0, 51, 51],
        [0, 51, 0],
        [51, 0, 25],
        [51, 0, 0],
        [160, 160, 160],
        [32, 32, 32],
        [160, 160, 160],
        [32, 32, 32],
        [204, 0, 204],
        [204, 0, 102],
        [204, 0, 204],
        [204, 0, 102],
        [255, 204, 229],
        [255, 153, 204],
        [255, 204, 229],
        [255, 153, 204],
        [133, 140, 197],
        [94, 102, 161]
    ])
