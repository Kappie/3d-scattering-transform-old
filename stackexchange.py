
    import numpy as np
    import pyculib.fft
    from scipy.fftpack import fftn, ifftn


    def convolution_gpu_naive(a, b):
        n_elements = np.prod(a.shape).astype(np.complex64)
        convolution = np.empty(a.shape).astype(np.complex64)
        a_fourier = np.empty(a.shape).astype(np.complex64)
        b_fourier = np.empty(a.shape).astype(np.complex64)
        pyculib.fft.fft(a, a_fourier)
        pyculib.fft.fft(b, b_fourier)
        a_fourier = a_fourier / n_elements
        pyculib.fft.ifft(a_fourier*b_fourier, convolution)
        return convolution


    def convolution_cpu(a, b):
        a_fourier = fftn(a)
        b_fourier = fftn(b)
        return ifftn(a_fourier*b_fourier)


    def identity_cpu(a):
        return ifftn(fftn(a))


    def identity_gpu(a):
        a_fourier = np.empty_like(a)
        result = np.empty_like(a)
        pyculib.fft.fft(a, a_fourier)
        pyculib.fft.ifft(a_fourier, result)
        return result / np.prod(a.shape)

    def print_absolute_differences_real_imag(a, b):
        print("average absolute difference of real part")
        print(np.average(np.abs(np.real(a - b))))
        print("average absolute difference of imaginary part")
        print(np.average(np.abs(np.imag(a - b))))


    def generate_random_arrays(x, y, z, seed=0):
        np.random.seed(seed)
        a_real = np.random.rand(x, y, z).astype(np.complex64)
        a_imag = np.random.rand(x, y, z).astype(np.complex64)
        b_real = np.random.rand(x, y, z).astype(np.complex64)
        b_imag = np.random.rand(x, y, z).astype(np.complex64)

        return a_real + 1j*a_imag, b_real + 1j*b_imag

    x = y = z = 128
    a, b = generate_random_arrays(x, y, z)

    cpu_result = convolution_cpu(a, b)
    gpu_result = convolution_gpu_naive(a, b)

    print_absolute_differences_real_imag(cpu_result, gpu_result)
    # average absolute difference of real part
    # 2.8885e-06
    # average absolute difference of imaginary part
    # 3.69549e-06
