import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load raw X data (snow melted + precipitation) - just element X1 of the loaded matlab object
raw_data = scipy.io.loadmat('data/utah/X_28x18.mat')['X1']

# Load mask - element ismap of the loaded matlab object
mask = scipy.io.loadmat('data/utah/mask_28x18.mat')['ismap']

# Apply (dot_product) mask on the raw data to generate masked data (X)
X = np.average(raw_data * mask, axis=(2, 3))

# Save the masked data
np.save('data/utah/X_masked.npy', X, allow_pickle=False, fix_imports=False)

# Load y data (stream flow) - element 'Q1' of the loaded matlab object
y = scipy.io.loadmat('data/utah/y.mat')['Q1']

# # Plot the data
# plt.plot(X)
# plt.plot(y)
# plt.show()

# Save the data to 
# Generate dates starting from Jan 1st, 1980
start_date = '1980-01-01'
dates = pd.date_range(start=start_date, periods=len(X), freq='D')

# Add an empty element at the beginning of y
y = np.insert(y, 0, None)

# Create a DataFrame
df = pd.DataFrame({'time': dates, 'X': X.flatten(), 'y': y.flatten()})

# Save the database
df.to_csv('data/utah/data.csv', sep=',', index=False, encoding='utf-8')