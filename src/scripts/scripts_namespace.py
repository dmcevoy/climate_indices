from multiprocessing import Array
import ctypes

shared_array = Array(ctypes.c_double, 1428 * 50, lock=False)
