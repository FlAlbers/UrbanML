import random

def generate_random_arrays():
    array1 = [random.randint(1, 10) for _ in range(10)]
    array2 = [random.randint(1, 10) for _ in range(10)]
    return array1, array2

y = generate_random_arrays()
type(y)