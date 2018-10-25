
# 1. (a) Z-Score Normalization or Standard Score Normalization

# Importing the libraries that we are using
import numpy.linalg as LA
from scipy import stats
import pandas as pd
import numpy as np

# Loading the input text dataset and reading it as csv using a pandas dataframe

csv = pd.read_csv('C:/Users/ebhavaniprasad/Desktop/magic04.txt', header=None)

print("data structure : ", type(csv))

print("data with labels")
print(csv.head(3))

# Removing class label attribute(not considering last column) from the dataset and storing it in a 'new_csv' object
new_csv = csv[csv.columns[:-1]]

# Taking the transpose of the input data
transp = new_csv.T
print("data without class labels")
print(new_csv.head(3))

# Storing the transposed data into a new object called 'df'
df = transp
print("transposed data")

print(df)

# Taking the sum along the rows in a transposed data 'df' and naming it as 'sum' and appending it to the same dataframe
df["sum"] = df.sum(axis=1)

print(df)

# Taking the 'sum' column into an object 'sum'
sum = df['sum']

print("the sum ")

print(sum)

# Removing the sum column from the transposed data frame 'df' and storing it in a 'df_new' data frame object
df_new = df[df.columns[:-1]]

print(df_new)

# Number of rows in an input data
n = len(df_new.columns)

print("N = ", len(df_new.columns))

# Calculating the mean for each column in a data frame
mean = sum / n

# Taking the transpose of the sum vector of size 10x1 and converts to 1x10
m_df = mean.to_frame().T
print("m_df")
print(m_df)
print(type(df_new))
# df_new.sub(m_df(axis=1), axis=0)

# Taking transpose and converting into a normal shape
Trans = df_new.T
# data in actual given dimension
print(Trans)

# Dimension of the data
print(Trans.shape)
# dimension of the mean column
print(m_df.shape)

# (Centered data) xi-mean
diff = pd.DataFrame(Trans.values-m_df.values, columns=Trans.columns)
DT = diff.T
print("x-mhu")
print(DT)
DTP = DT.pow(2)

# (xi- mean) square

print(DT.pow(2))

# Appending the (xi-mhu)square column to dataframe
DTP["sumsquare"] = DTP.sum(axis=1)

print(DTP)

# Storing the 'sumsquare' column to an 'ss' object
ss = DTP['sumsquare']

# Removing the sigma(xi-mean)square column from the data frame 'DTP'
DTP_new = DTP[DTP.columns[:-1]]
print(ss)

# Calculating the variance
print("Variance")
variance = ss/n
print(variance)


# Calculating the Standard deviation
sd = variance.pow(1./2)   # Square root of variance

print("Standard Deviation")
print(sd)

# Calculating the Z- Score values for data frame
znor = DT.div(sd, axis='index')

print("z-nor")

print(znor)


print("library scipy")
print(stats.zscore(df, axis=1, ddof=1))


# Comparing mean using numpy built-in function and manually calculated mean
print("MEAN USING NUMPY BUILT-IN FUNCTION")
print(np.mean(csv))
print("MANUALLY CALCULATED MEAN")
print(mean)

# Comparing the variance using built-in function and manual calculation
print("VARIANCE USING NUMPY BUILT-IN FUNCTION", np.var(csv))
print("VARIANCE USING MANUAL CALCULATION", variance)

# Final Z-normalized data
print("MANUALLY CALCULATED Z-SCORE")

znormal = znor.T
print(znormal)
csv3 = csv
csv5 = new_csv

# Converting Pandas Data Frame to NumPy ndarray
csv4 = csv3.values

# print("type of csv 4", type(csv4))

df_zscore = (csv5 - csv5.mean())/csv5.std()

print("Z-SCORE USING LIBRARY FUNCTION")
print(df_zscore)

znormal2 = znormal.copy()

# Mean of the Z- Score Normalized Dataset = 0
print("Mean of the Z-Score Normalized Dataset")
print(znormal2.values.mean())

# Standard Deviation of the Z- Score Normalized Dataset = 1
print("Standard Deviation of the Z-Score Normalized Dataset")
print(znormal2.values.std(ddof=1))


# ----------------------------------------------------------------------------------------------------------------
# 1. (b) Sample Covariance Matrix

z_score2 = znor

z_score = znor

# Calculating the Z-Score each row and appending it to a 'z_score' data frame
z_score["Z- score sum"] = z_score.sum(axis=1)

# Z-score data frame after appending the sum of Z-scores column
print('z-score mean')
print(z_score)

# Storing sum of z-scores column to a 'z_sum' object
z_sum = z_score['Z- score sum']

print("The z-score sum ")

print(z_sum)

# Removing the 'z_sum' column from 'z_score' data frame
z_score_new = z_score[z_score.columns[:-1]]

print(z_score_new)

# Calculating the Z-score mean
z_mean = z_sum/n

print('z-mean')

# converting the series to data frame
z_mean_df = z_mean.to_frame().T
print(type(z_mean_df))
print(z_mean_df.shape)
print(z_score_new)

z_data = z_score_new.T

# zxi-mean
diff2 = pd.DataFrame(z_data.values-z_mean_df.values, columns=Trans.columns)
zD = diff2

print('Z-Centered Data')

print(zD)

# Transpose of the Z-Centered data
zDT = zD.T

print(zDT)

zD2 = zD
zDT2 = zDT

# sigma  = pd.DataFrame(zDT.values*zD.values)

#sigma = zDT * zD

# Converting the data frame to an ndarray
matri1 = zD2.values
matri2 = zDT2.values

print(type(matri1), "", matri1.shape)
print(type(matri2), "", matri2.shape)

# multiplying two ndarrays of order 10x19020 * 19020x10
prod = np.matmul(matri2, matri1)

# Normalizing the Covariance Matrix by N
cov_mat = prod/n

print("covariance matrix shape ", cov_mat.shape)

print("COVARIANCE MATRIX USING MANUAL CALCULATION ")
print(cov_mat)


print(z_data.shape)

z_data2 = z_data

# Covariance of the z-score data using built-in function cov()
result = z_data2.cov()

print("Dimension of the Covariance matrix : ", result.shape)

print("COVARIANCE MATRIX USING DATA FRAME COV() BUILT-IN FUNCTION ")
print(result)


# ---------------------------------------------------------------------------------------------------------------------


# 1.  (c)  Dominant Eigenvalue and Eigenvector of the Covariance Matrix using Power Iteration Method

# Deep copying the cov_mat to a cov_mat_new
cov_mat_new = np.copy(cov_mat)

# Dimension of the covariance matrix 'd' or number of columns in it.

d = cov_mat_new.shape[1]

# Starting vector - x0 (It generates the random vector of size 10x1
x = np.random.rand(d, 1)
# print(type(x))
# print(type(cov_mat_new))

# Boolean flag to check and stop for the condition
flag = True

count = 0
while(flag==True):
    y = cov_mat_new.dot(x)
    max_val = np.amax(abs(y))       # Takes the absolute maximum value in a vector 'x'
    n = y / max_val                 # Re-scaling the 'y' such that maximum value in the 'y' will be 1 for next iteration
    p = n-x                         # Taking Difference of the scaled vector of current iteration and previous

    if(count<100):
        count = count + 1

    if(LA.norm(p)<0.000001):        # Checking the norm of the difference of vector is less than the threshold 0.000001
        flag = False                # It falls below the threshold limit so now stopping condition flag = False

    x = np.copy(n)                  # Assigns the current iteration scaled vector as xi-1 for next iteration


print("Iterations took for convergence : ", count)

# Unit normalization
norm_eigen = LA.norm(n)

# Unit Eigenvector of length = 1
unit_vect = n/norm_eigen

print("\n THE DOMINANT EIGEN VALUE FOR COVARIANCE MATRIX USING MANUAL CALCULATION")
print(max_val)

print("\n THE DOMINANT EIGEN VECTOR FOR COVARIANCE MATRIX USING MANUAL CALCULATION")
print(unit_vect)

print("\n The length of the Final Eigen Vector : ", LA.norm(unit_vect))

# Using numpy linalg.eig built-in function to verify answers
EValue, EVector = LA.eig(cov_mat_new)

print("\n EIGEN VALUES FOR COVARIANCE MATRIX USING BUILT-IN LIBRARY FUNCTION ")
print(EValue)

print("\n EIGEN VECTOR FOR COVARIANCE MATRIX USING BUILT-IN LIBRARY FUNCTION ")
print(EVector)

# ---------------------------------------------------------------------------------------------------------------------

# 1. (d)

# Sorting the Eigenvalues for finding the two dominant Eigen vectors of the covariance matrix

pos = EValue.argsort()[::-1]               # Sorts in a descending order and takes the indexes of the eigenvalues
EValue1 = EValue[pos]                    # sorts the eigenvalues in a descending order
EVector1 = EVector[:, pos]               # Sorts the eigenvector in a descending order

print("Sorted Eigenvalues and respective Eigenvectors")
print("\n Before sorting the Eigenvalue")
print(EValue)
print("\n After sorting the Eigenvalue")
print(EValue1)
print("\n Before sorting the Eigenvector")
print(EVector)
print("\n After sorting the Eigenvector")
print(EVector1)

# First Two Dominant Eigenvectors of Covariance Matrix
dom = 2   # enter the number of dominant eigenvectors you want
print("\n First Two Dominant Eigenvectors of Covariance Matrix")
dom_evect = EVector1[:, :dom]

print(dom_evect)

print(dom_evect.shape)

print(type(dom_evect))

# Projection of a data point onto first two dominant eigenvectors

# Deep copying the Z-normalized score dataset
zscore_copy = z_score_new.copy(deep=True)

print("zscore copy")
print(zscore_copy)

zscore_copy_transp = zscore_copy.T

zee_data = zscore_copy_transp.values

print(zee_data.shape)


# Projection of data points on the subspace spanned by two dominant eigenvectors

proj_data = zee_data.dot(dom_evect)                       # Multiplying Z-score data * dominant eigen vector

print("data structure : ", type(proj_data), " shape : ", proj_data.shape)

print("\n projection of data points spanned by 2-dominant eigenvector")
print(proj_data)

# Variance of the data points in the projected subspace


eigen_2 = EValue1[0:dom]    # Taking two dominant eigenvalues

esum = eigen_2.sum()        # Sum of two dominant eigenvalues = variance of the data points spanned on two eigenvectors
print("\n THE VARIANCE OF DATA POINTS ON PROJECTED SUBSPACE : ", esum)

# --------------------------------------------------------------------------------------------------------------------

# 1. (e) Covariance matrix in its Eigen-decomposition form

V = EVector.copy()           # Deep copying the EigenVectors

print("edt")
print(V.shape)

# Transpose of a eigenvector matrix
W = V.T
print(W.shape)

L = np.diagflat(EValue)

print(L)

# Eigen-decomposition matrix multiplication
c1 = L.dot(W)
C = V.dot(c1)


print("\n COVARIANCE MATRIX IN EIGEN-DECOMPOSITION FORM UVUT")
print(C)

# C matrix and Covariance Matrix should be same, AX=LX form
print("\n COVARIANCE MATRIX ")

print(cov_mat_new)

# --------------------------------------------------------------------------------------------------------------------

# 1. (f) Subroutine that implements PCA Algorithm

# Deep copying the Z-Score data for PCA
D = z_score_new.copy(deep=True)


def PCA(D, threshold):
    D2 = D.T
    print(D2)

    D3 = D2.values  # Converting DataFrame to ndarray
    D4 = D.values
    N = D3.shape[0]  # Number of rows in data

    # Computing the Covariance Matrix
    sigma1 = D4.dot(D3)
    sigma2 = sigma1 / N
    print(sigma2.shape)  # Covariance Matrix

    # Finding the EigenValues and EigenVectors for the Covariance Matrix (sigma2)

    eigenval, eigenvect = LA.eig(sigma2)

    # Sorting the Eigenvalues that are dominant and associated eigenvectors
    indexes = eigenval.argsort()[::-1]         # Sorting the eigenvalues in a descending order and taking indexes
    eigenval1 = eigenval[indexes]              # Sorted eigenvalues in descending order
    eigenvect1 = eigenvect[:, indexes]         # Sorting corresponding eigenvectors

    print("Unsorted eigenvalues ", eigenval)

    print("Dominant eigenvalues ", eigenval1)

    total_variance = eigenval.sum()           # Sum of all eigenvalues in a Covariance matrix is a total variance

    print("Total Variance ", total_variance)

    S = len(eigenval1)
    print("S", S)
# Iteration counter that tell us the number of eigenvalues that summed together to preserves a 95% of variance of data
    eig_count = 0

    flag2 = True

    for ele in range(S):
        # It tell us the percentage of the variance of data that each eigenvector preserves
        percentage = [(i / total_variance) * 100 for i in eigenval1]
        # Cummulates the percentage of data  of the each of the eigenvector preserves
        position = np.cumsum(percentage)
        # Stopping condition and choosing dimensionality of eigenvector that preserves 95% of data
        if (position[ele] >= threshold):
            break
            flag2 = False
            ele

        eig_count = eig_count + 1

    eig_count2 = eig_count + 1           # Adding 1 since for loop itearation starts at 0

    print("The number of eigevectors that preserves the variance of %d" %threshold, "% is",  eig_count2)

    print("eigen vector ", eigenvect1)

    eigenvect1_copy = eigenvect1.copy()
    eig_transpose = eigenvect1_copy.T

    # EigenVector for Ur as a new basis vector

    # Taking the dominant eigen vector that cover 95% of the variance of the data
    Ur = eig_transpose[0:eig_count2, :]

    Ur2 = Ur.T

    # print(Ur2.shape)
    a1 = z_score_new.copy()
    a2 = a1.T

    a3 = a2.values                            # Converting dataframe to an ndarray
    # Reduced Dimensionality data
    Reduced_data2 = a3.dot(Ur2)

    # Enter the number of datapoints want to print using the Principal Vectors as a new Basis Vector
    m = 10
    Ten_datapoint = Reduced_data2[0:m, :]

    print('COORDINATES OF THE FIRST TEN DATA POINTS')

    return Ten_datapoint, eigenval1, eig_count2, Reduced_data2



# Percentage of Variance of data want to preserve using the Principal Components Vectors
alpha = 95

Reduced_data, eigenvalue1, eigencount3, Reduced_data3 = PCA(D, alpha)

# COORDINATES OF THE FIRST TEN DATA POINTS
print(Reduced_data)

# print(eigenvalue1)


# ---------------------------------------------------------------------------------------------------------------------

# 1. (g)  Covariance of the Projected data points


# Computing Covariance of the data points spanned on the principal vectors
covariances5 = np.cov(Reduced_data3, bias=True, rowvar=False)

print("\n The Covariance Matrix of the Projected data points\n ")

print(covariances5)

# Finding the trace of the Covariance matrix of the projected data points
Trace_covariance = np.trace(covariances5)

print(Trace_covariance)

# Sum of eigenvalues corresponding to the principal vectors on which the data is projected
princ_evalue = eigenvalue1[0:eigencount3, ]

sum_principal = princ_evalue.sum()

print("COVARIANCE OF THE PROJECTED DATA POINTS : ", Trace_covariance)

print("SUM OF THE EIGENVALUES CORRESPONDING TO THE PRINCIPAL VECTORS ON WHICH DATA IS PROJECTED : ", sum_principal)





