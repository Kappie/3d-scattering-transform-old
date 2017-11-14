import numpy as np

from numpy.lib.format import open_memmap


filename = r"D:\test_memmap.dat"
a, b, c, d = (25, 100, 100, 4)
shape = (a, b, c, d)
dtype = np.float32
# result = open_memmap(filename, mode="w+", dtype=dtype, shape=shape)


# for i in range(a):
#     if i % 10 == 0:
#         print("working on {}".format(i))
#     data = np.random.rand(b, c, d).astype(dtype)
#     result[i, :, :, :] = data

result = open_memmap(filename)
print(result)
