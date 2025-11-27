import numpy as np

def statistics (name_array):  #Function to calculate the average, median, min and max
    mean = np.mean(name_array)
    median = np.median(name_array)
    min = np.min(name_array)
    max = np.max(name_array)

    print(f"The average is {mean:.2f}")
    print(f"The median is {median:.2f}")
    print(f"The minimum is {min:.2f}")
    print(f"The maximum is {max:.2f}")
    print()
    return mean, median, min, max
