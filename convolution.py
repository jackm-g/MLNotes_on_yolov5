import numpy as np
import scipy

input_matrix = np.array([[1, 5, 4, 2, 9],
                        [3, 7, 6, 0, 1],
                        [2, 4, 7, 5, 4],
                        [4, 1, 2, 4, 3],
                        [4, 5, 4, 6, 4]])

filter = np.array([[1, 0 , -1],
                    [1, 0 , -1],
                    [1, 0 , -1]])

result = scipy.signal.correlate(input_matrix, filter, mode='valid')

#print(result)



# Grayscale image example

# 6x6
image_matrix = np.array([[0, 0, 0, 10, 10, 10],
                        [0, 0, 0, 10, 10, 10],
                        [0, 0, 0, 10, 10, 10],
                        [0, 0, 0, 10, 10, 10],
                        [0, 0, 0, 10, 10, 10],
                        [0, 0, 0, 10, 10, 10]])

res2 = scipy.signal.correlate(image_matrix, filter, mode='valid')

print(res2)