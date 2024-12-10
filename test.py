import numpy as np

dict = {1: "Bali", 2: "Africa"}

result = dict[np.random.choice([1, 2])]
print(result)
