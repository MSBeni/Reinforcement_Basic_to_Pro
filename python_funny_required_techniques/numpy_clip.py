# ref: https://www.geeksforgeeks.org/numpy-clip-in-python/

import numpy as np

# numpy.clip() function is used to Clip (limit) the values in an array.
# Given an interval, values outside the interval are clipped to the interval edges. For example, if
#     an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.

#########################################################################
in_array = [1, 2, 3, 4, 5, 6, 7, 8]
print("Input array : ", in_array)

out_array = np.clip(in_array, a_min=2, a_max=6)
print("Output array : ", out_array, "\n")
##########################################################################

#########################################################################
a_in = np.random.random((2,3))*100
print("Second Input array : ", a_in , "\n")
print("Output array 2nd: ", np.clip(a_in, a_min=10.0, a_max=80.00), "\n")
#########################################################################

#########################################################################
in_array_3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Input array 3 : ", in_array_3)

out_array_3 = np.clip(in_array_3, a_min=[3, 4, 1, 1, 1, 4, 4, 4, 4, 4], a_max=9)
print("Output array 3 : ", out_array_3, "\n")
#########################################################################


#########################################################################
in_array_4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Input array 4 : ", in_array_4)

out_array_4 = np.clip(in_array_4, a_min=[3, 4, 1, 1, 1, 4, 4, 4, 4, 4],
                    a_max=[10, 8, 10, 10, 10, 10, 9, 4, 8, 4])
print("Output array 4 : ", out_array_4, "\n")
#########################################################################