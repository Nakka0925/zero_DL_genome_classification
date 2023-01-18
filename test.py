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
                self.images.append(path)
                self.labels.append(cls)

                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images, dtype=np.float32)
                    targets = np.asarray(self.labels, dtype=np.float32)
                    self.reset()
                    yield inputs, targets


test = ImageDataGenerator()

a = list(range(1,101))
b = list(range(1,101))

for idx, a in enumerate(test.flow_from_directory(a,b), start=1):
    print(idx)
    if (idx == 4):
        print("=========================")
    if (idx == 6): break
