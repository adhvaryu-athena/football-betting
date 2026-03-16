import numpy as np
import pandas as pd

df = pd.DataFrame(np.array([1, 2, 3]))  

df.to_csv("test.csv", index=False)