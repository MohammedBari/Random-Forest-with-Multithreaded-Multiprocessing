“Random Forest with Threading and Multiprocessing”

Undergraduate Research Project - Cal Poly Pomona

Mohammed Bari

Professor John Korah


INTRODUCTION: 

This project aims to identify efficient solutions for the creation of species distribution models from satellite images of regions. These species distribution maps are based on classification tree (CT) algorithms, which aims to produce categorical predictions. 
Employing CT algorithms through the random forest method, the prediction models are created with presence only data. Species distribution models are predictive models that can estimate the relationships between environmental characteristics of an area and the species occurrences within an area. With machine learning, these models can be created with more efficient functions and results, such as being able to use certain algorithms to arrive at an output. 

This project focuses on Random Forest algorithms for use with species distribution models, in addition to a multithreaded parallelized implementation, a potential tool for the large amounts of data that are used. The strengths and drawbacks will be explored RF working with data sets to produce occurrence data, and its performance will be evaluated across various parallelized implementations.


IMPLEMENTATION:

The code for the implementation is done through python, using the Sci-Kit learn library. The built in “RandomForestClassifier” and the train_test_split module of the library contains training functions that are employed in this implementation. The RandomForestClassifier, is described as “a meta estimator” that uses sub-samples of the given dataset to fit decision trees, and then averages to control over-fitting and improve accuracy. The parameter max_samples controls the sub-sample size with its  default Boolean bootstrap=True. A false setting would mean the entire dataset is used.


DATASETS USED: 

The data used with this experiment are ARCGIS files obtained by the CPP GIS department, containing 19 bioclimatic variables for species. The occurrence data for the Malagasy giant chameleon, or the Furcifer oustaleti animal is used for training data. The data across several files, include 7357 pointid’s that each have their own 19 bioclimatic variables. The pointid are a field made and added to the test and training sets meant to act as a Boolean value. The actual pointid’s correspond to a latitude and longitude with their own 19 variables, and for this example, they were used as a stand in for the coordinate points, and will have their own Exists counterpart, which is a Boolean value denoting whether a species would exist at a pointid.


PARALLELIZED IMPLEMENTATION:

The main structure of both multiprocessing and multithreading here is that a dataset is divided into subsets, which are then fed into a function that fits and runs prediction. 
The creation of the sets are done manually and before the function executes, so the actual model functionality occurs in threads or processes in parallel.
In creating the methods, the multiprocessing portion is implemented with the multiprocessing library, using the lines multiprocessing.Process(target=runPredictions1) per each process used. 
The processes are started and joined, allowing them to run almost simultaneously. Each random forest instance is tied to a process that executes separately, and the entire program is completed when the last process finishes. 
In addition, sci-kit allows for built in multiprocessing into the RandomForestClassifier(n_estimators = 1500, n_jobs = 2) command, where assigning value to the n_jobs can allow for each forest to be processed separately.
For the multithreaded implementation, each thread is created, with the method for creating sets, fitting, and then predicting executing in each thread. The threaded implementation uses the manual feature assignment method, and also uses 1500 trees calculate its results.
In addition, a combination of both multithreading and multiprocessing is implemented. The threaded portion has the entirety of the runpredictions function handled within it, while inside the function the n_jobs parameter is set to a value that allows for the RandomForestClassifier method to run the fitting and predictions in parallel dependent on the value passed.
