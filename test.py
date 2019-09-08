# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:18:37 2019

@author: Fede
"""

import numpy as np
import matplotlib.pyplot as plt
import binvox_rw
with open('chair.binvox', 'rb') as f:
     m1 = binvox_rw.read_as_3d_array(f)

print (m1.dims)



round(m1.scale,3)

m1.translate

with open('chair_out.binvox', 'wb') as f:
     m1.write(f)
with open('chair_out.binvox', 'rb') as f:
     m2 = binvox_rw.read_as_3d_array(f)

m1.dims==m2.dims

m1.scale==m2.scale
m1.translate==m2.translate
np.all(m1.data==m2.data)

with open('chair.binvox', 'rb') as f:
     md = binvox_rw.read_as_3d_array(f)
with open('chair.binvox', 'rb') as f:
     ms = binvox_rw.read_as_coord_array(f)

data_ds = binvox_rw.dense_to_sparse(md.data)
data_sd = binvox_rw.sparse_to_dense(ms.data, 32)
np.all(data_sd==md.data)

plt.figure()
plt.imshow(data_sd[1])

# the ordering of elements returned by numpy.nonzero changes with axis
# ordering, so to compare for equality we first lexically sort the voxels.
np.all(ms.data[:, np.lexsort(ms.data)] == data_ds[:, np.lexsort(data_ds)])
