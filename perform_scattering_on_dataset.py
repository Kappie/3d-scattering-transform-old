import numpy as np
import datetime

from scattering_transform import apply_scattering_transform_to_dataset


def generate_output_location(js, J, L, sigma):
    BASE_NAME = "results"
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "js{}-{}_J{}_L{}_sigma{}_{}.dat".format(js[0], js[-1], J, L, sigma, datetime_string)
    return os.path.join(BASE_NAME, file_name)


if __name__ == '__main__':
    DATASET_PATH = r"F:\GEERT\DATASET_NUMPIFIED\dataset.npy"
    LABELS_PATH = r"F:\GEERT\DATASET_NUMPIFIED\labels.npy"
    AFFECTED = 1
    UNAFFECTED = -1

    dataset = np.load(DATASET_PATH, mmap_mode="r")
    labels = np.load(LABELS_PATH)

    js = [0, 1, 2, 3]
    J = js[-1]
    L = 3
    sigma = 3
    output_location = generate_output_location(js, J, L, sigma)
    
    apply_scattering_transform_to_dataset(dataset, js, J, L, output_location, sigma=sigma)
