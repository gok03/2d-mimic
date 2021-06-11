import cv2

def draw_line(img, coordinates):
    """
    Args:
        img: image on which you want to draw
        coordinates: list of list [[1, 2], [3, 4], [5, 6]]
    """
    for i in range(len(coordinates)):
        cv2.line(img, coordinates[i - 1], coordinates[i], 2)

    return img

def draw_shades(img, coordinates, relations):
    """
    Args:
        img: image on which you want to draw
        coordinates: list of list [[1, 2], [3, 4], [5, 6]]
        relations: relations between coordinates
    """
    pass

def draw_polygons(img, coordinates):
    pass




