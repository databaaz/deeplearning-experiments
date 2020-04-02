# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from dlframework.preprocessing import SimplePreprocessor
from dlframework.datasets import SimpleDatasetLoader
from imutils import paths
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type =int, default= 1, 
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default = -1,
                help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())


# grab the list of images that we’ll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk, and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data , labels) = sdl.load(imagePaths, 500)
data = data.reshape(data.shape[0], 3072)

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes/(1024*1000.0)))


# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of the data for 
# training and the remaining 25% for testing
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.75, random_state = 42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
knn = KNeighborsClassifier(n_neighbors=args["neighbors"],
                            n_jobs = args["jobs"])
knn.fit(X_train, Y_train)

print(classification_report(Y_test, knn.predict(X_test),
                    target_names = le.classes_))