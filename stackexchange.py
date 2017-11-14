
    coefficients = np.memmap(
        output_location, dtype=np.float32, mode="w+",
        shape=(n_samples, number_of_transforms, width, height, depth))

    for n in range(n_samples):
        image = images[n]
        coefficients_sample = transform(images[n])
        coefficients[n, :, :, :, :] = coefficients_sample
