import cv2

import numpy as np


class ImageDataGenerator(object):
    def __init__(self, rescale=None):
        self.rescale = rescale
        self.reset()

    def reset(self):
        self.images = []
        self.labels = []

    def flow_from_directory(self, img_path, classes, batch_size=32):
        while True:
            for path, cls in zip(img_path, classes):
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.images.append(1 - np.asarray(img, dtype=np.float16) / 255)
                self.labels.append(cls)

                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images, dtype=np.float16)
                    targets = np.asarray(self.labels)
                    inputs = inputs.reshape(-1, 1, 192, 192)
                    self.reset()
                    yield inputs, targets