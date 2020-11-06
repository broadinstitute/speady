import cython
from cython import Py_ssize_t
import numpy as np
cimport numpy as cnp
from numpy cimport (ndarray,
                    NPY_INT64, NPY_INT32, NPY_INT16, NPY_INT8,
                    NPY_UINT64, NPY_UINT32, NPY_UINT16, NPY_UINT8,
                    NPY_FLOAT32, NPY_FLOAT64,
                    NPY_OBJECT,
                    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                    uint32_t, uint64_t, float32_t, float64_t)
cnp.import_array()
cdef float64_t NaN = <float64_t>np.NaN
from libc.math cimport fabs, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def optim_cy_pearson(ndarray[float64_t, ndim=2] x, ndarray[float64_t, ndim=2] y):
    cdef:
        Py_ssize_t i, x_rows, x_cols, y_rows, y_cols, nobs, x_i, y_i
        Py_ssize_t min_points = 1
        ndarray[uint8_t, ndim=2] x_mask, y_mask
        ndarray[float64_t, ndim=2] result

        float64_t sum_x, sum_y, sum_xy, sum_xx, sum_yy, mean_x, mean_y, divisor, x_v, y_v

    x_rows, x_cols = (<object>x).shape
    y_rows, y_cols = (<object>y).shape

    result = np.empty((x_cols, y_cols), dtype=np.float64)
    x_mask = np.isfinite(x).view(np.uint8)
    y_mask = np.isfinite(y).view(np.uint8)

    for x_i in range(x_cols):
        for y_i in range(y_cols):
            nobs = sum_xx = sum_yy = sum_x = sum_y = sum_xy = 0
            for i in range(x_rows):
                if x_mask[i, x_i] and y_mask[i, y_i]:
                    nobs += 1
                    sum_x += x[i, x_i]
                    sum_y += y[i, y_i]
            if nobs < min_points:
                result[x_i, y_i] = NaN
            else:
                mean_x = sum_x / nobs
                mean_y = sum_y / nobs
                for i in range(x_rows):
                    if x_mask[i, x_i] and y_mask[i, y_i]:
                        x_v = x[i, x_i] - mean_x
                        y_v = y[i, y_i] - mean_y
                        sum_xy += x_v * y_v
                        sum_xx += x_v * x_v
                        sum_yy += y_v * y_v
                divisor = sqrt(sum_xx * sum_yy)
                if divisor != 0:
                    result[x_i, y_i] = sum_xy / divisor
                else:
                    result[x_i, y_i] = NaN

    return result
