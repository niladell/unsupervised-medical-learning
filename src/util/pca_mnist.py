import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.util.dcm_manipulation import greyscale_plot, static_plotting, interactive_plotting
import pandas as pd


mnist = sklearn.datasets.load_digits()

train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)
mnist_rescaled = scaler.transform(mnist.data)

# Make instance of model
pca = PCA(.95)
pca.fit(train_img)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# Let's use all the data for plotting
mnist_transformed = pca.transform(mnist_rescaled)


# Visualizing the principal components:
for i in range(15):
    greyscale_plot(pca.components_[i, :].reshape(8,8))

# Plotting the transformed datapoints
columns = []
for i in range(1, pca.n_components_ + 1):
    string = 'principal component ' + str(i)
    columns.append(string)

principalDf = pd.DataFrame(data=mnist_transformed, columns=columns)
static_plotting(principalDf, mnist.target, title='')







