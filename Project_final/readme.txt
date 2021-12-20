(a) code for data preprocessing
one hot encoding technique (OHE):
	Final_report_Data_prep_OHE.py
	remove the quote for training and test data respectively, the preprocessed results will automatically generated.
	train_p.csv (25000 rows, 105 columns)
	test_p.csv (23842 rows, 104 columns)
Bin Counting technique (BC):
	Final_report_Data_prep_bincount.py
	remove the quote for training and test data respectively, the preprocessed results will automatically generated.
	train_bin_count.csv (25000 rows, 105 columns)
	test_bin_count.csv (23842 rows, 104 columns)
########################################################################
(b) logistic regression
pf_logistic_regression.py

# selecting the two groups of the data file in (a) repectively to analyze the data preprocessed by OHE, and BC, respectively.
change the file path: in line 16 and 31.
cross correlation: 
	remove the quote in line 233 and 269, run the file
determining the best variance: 
	remove the quote in line 271 and 297, run the file
plot the training data: 
	remove the quote in line 300 and 305, run the file
compute the confusion matrix:
	remove the quote in line 307 and 320, run the file
predicting the test data:
	using line 326 and 327 for BC and OHE, respectively, and line 343 and 349, respectively to store the predictions
#########################################################################
(c) code for neural networks
# ===================================
pf_NN.py: the neural networks algorithm for the final project.
part I) plot the training error 
a) OHE case
import the OHE file (train_p, test_p)
set the epoch = 1 , in line 132
layers = 3, in line 114
width = 105, in line 115
input width = 104 , in line 120
remove the quote in line 166, 170, 186, 196
Then, run the file, it will automatically plot the training errors for the training dataset.
=====================================
b) bin counting case
import the bin counting file (train_bin_count, test_bin_count)
set the epoch = 1 , in line 132
layers = 3, in line 114
width = 15, in line 115
input width = 14 , in line 120
remove the quote in line 166, 170, 186, 196
Then, run the file, it will automatically plot the training errors for the training dataset.
#======================================
part II) compute the training errors under different layers and width.
remove the quote, run the file

part III) predict the test data.
remove the quote, and use the provided dataset to run the file 


