import os
import SimpleITK as sitk
import numpy as np


def convert_to_image(path):
    array = np.load(path)["arr_0"]
    array = np.swapaxes(array, 0, 2)
    return sitk.GetImageFromArray(array)


INPUT_FOLDER = r"F:\GEERT\transforms"
OUTPUT_FOLDER = r"F:\GEERT\transforms_nii"
OUTPUT_FORMAT = ".nii"


for file_name in os.listdir(INPUT_FOLDER):
    print("Working on {}.".format(file_name))
    input_path = os.path.join(INPUT_FOLDER, file_name)
    output_path = os.path.join(OUTPUT_FOLDER, file_name) + OUTPUT_FORMAT
    image = convert_to_image(input_path)

    sitk.WriteImage(image, output_path)
