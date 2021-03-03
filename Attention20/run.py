import os
import sys
import time

# [0.01, 0.05, 0.1, 0.5, 1, 5, 10]

# a_lst = [0.1, 1, 5, 10, 20, 50, 100]
# b_lst = [0.1, 1, 5, 10, 20, 50, 100]

for i in range(200):
    command = "python saliency.py {}".format(i)
    os.system(command)
# for a in a_lst:
#     command = "python FIND_PARA.py {} {} 5000".format(a, 10)
#     os.system(command)
#
# for b in b_lst:
#     command = "python FIND_PARA.py {} {} 5000".format(10, b)
#     os.system(command)