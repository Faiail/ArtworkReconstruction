from enum import Enum
import numpy as np
import cv2
from PIL import Image

class MaskGenerator:

    # Class Constructor
    def __init__(self):
        self.probas = []            # list of the given probabilities
        self.draw_methods = []      # list of possible DrawMethods

        # can be configured
        line_proba = 1/3
        irregular_proba = 1/3
        mixed_proba = 1/3

        if line_proba > 0:
            self.probas.append(line_proba)
            self.draw_methods.append(DrawMethod.LINE)

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            self.draw_methods.append(DrawMethod.IRREGULAR)

        if mixed_proba > 0:
            self.probas.append(mixed_proba)
            self.draw_methods.append(DrawMethod.MIXED)

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()


    # Allows to call the class as a function
    def __call__(self, img, variants_n=5):
        masks = []

        # Get image shape
        (h, w, _) = np.shape(img)

        for _ in range(variants_n):
            # Randomly choise of the DrawMethod to use
            kind = np.random.choice(len(self.probas), p=self.probas)
            method = self.draw_methods[kind]
            masks.append(make_random_irregular_mask( shape = (h, w), draw_method = method ))

        return masks




class DrawMethod(Enum):
    LINE = 'line'
    IRREGULAR = 'irregular'
    MIXED = 'mixed'


def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=3, max_times=6, draw_method=DrawMethod.LINE):

    draw_method = DrawMethod(draw_method)
    height, width = shape
    mask = np.zeros((height, width), np.uint8)  # Convert to grayscale image
    times = np.random.randint(min_times, max_times + 1)

    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)

        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)

            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle

            length = 10 + np.random.randint(max_len)
            thickness = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)

            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, thickness)

            elif draw_method == DrawMethod.IRREGULAR:
                radius = np.random.randint(1, 15)
                irregularity = np.random.uniform(0, 10)
                contour = cv2.ellipse2Poly((start_x, start_y), (radius, radius), 0, 0, 360, 10)
                contour += (np.random.randn(*contour.shape) * irregularity).astype(np.int32)
                cv2.fillPoly(mask, [contour], 255)

            elif draw_method == DrawMethod.MIXED:
                k = np.random.randint(10)
                if k < 5:
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, thickness)
                else :
                    cv2.circle(mask, (start_x, start_y), radius=thickness, color=255, thickness=-1)

            start_x, start_y = end_x, end_y

    return Image.fromarray(mask)