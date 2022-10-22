## Dry Bean Class prediction using Machine Learning Techniques.
### Motivation to solve the Problem
Dry Beans are one of the most consumed edible legumes in the world. Seed quality affects the crop production and revenue.  
It is hence essential to classify dry bean seeds for optimal production and max revenue as well as for sustainable farming.

### Dataset and Publications Used
- The data is obtained from the following   [`dataset`](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset).
- The motivation for the problem is sourced form the following [ `paper`](https://www.sciencedirect.com/science/article/abs/pii/S0168169919311573?via=ihub). This paper helped me to understand the underlying the scientific basis for this problem.

### Goals of the Project
This clearly is a `multi class classification problem` where we need to classify a Dry bean seed into one of the six classes from ***Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira.***  
1. Data Preprocessing
2. Performing EDA on the given Dataset to analyse the data better.
3. Using `TSNE` to reduce the dimensions of the data and plot the scatter plot. This will help us to comment on the seperability of the data.
4. Predicting the class using Sklearn's Gaussian and Multinomial Naive Bayes Model.
5. Using PCA for reducing the dimensions of the data and analyse the performance.
6. Plot ROC and AUC for the data using logistic regression of Sklearn.
7. Preict the class using LogisticRegression of Sklearn and compare performance betweeen two Naive Bayes Models.

### File Structure  
- `Dry-Bean-Predictor.ipynb` -> Jupyter Notebook with ML Model  
- `Dry_Bean_Dataset.xlsx` -> Dataset
## Methodology and Results Obtained.
## Part 1.
- Since we are making use of Libraries here, we need to import all the relevant libraries like numpy, matlplotlib, seaborn and pandas.
We then make a pandas dataframe from the given CSV Dataset file.
We then make a histogram and line plot according to the ‘CLASS” column of the given dataset.  
![image](https://user-images.githubusercontent.com/76804249/197333397-c78b59cc-c9e5-4666-9570-adf448b82594.png)

- We now plot the histograms of all other classes with respect to features
![image](https://user-images.githubusercontent.com/76804249/197333427-d43d0c84-4ce2-4dd1-8f6f-d51d0b20399e.png)

### **Analysis from the Data**
- Most Number of beans have a major axis length of around 250 and minor axis length of around 190.
- Area of around 42000 is the most common area
- Most number of beans belong to the class DERMASON and the least number of beans to the class BOMBAY

## Part 2
- Here we need to perform and Explanatory Data Analysis.
We use df.info() to get the details about the dataset like number of columns, value types etc and df.describe() to get the details like count, average, maximum and minimum etc for the all the columns. We also use df.shape() to get the number of rows which is 13611 and number of columns which are 17 for our dataset.
Now first we plot the histogram for each column. This helps us better visualise the frequency of values present in a column.  
![image](https://user-images.githubusercontent.com/76804249/197333554-fdbe2ee9-983c-4497-b11f-9bd4d630898b.png)

- Now we plot a scatterplot between Area and Perimeter which better helps understand the relationship between the two and what area and perimeter is common in a class.
We observe that area is directly related to the perimeter and that the BOMBAY class tends to have the highest area and perimeter.  
![image](https://user-images.githubusercontent.com/76804249/197333587-22dbab8d-bf16-4fea-a63a-b56dd64ba369.png)

- We now make the Boxplots for Area and perimeter to check for the outliers in the data. We find there are a lot of outliers in the data.  
![image](https://user-images.githubusercontent.com/76804249/197333616-6aea0422-dc67-44c6-a317-56e3bf7a5370.png)

- We now plot a pairplot to given the pairwise correlation between of the columns using 
sns.pairplot() function. A high positive correlation means that the columns are highly positively related and a high negative correlation means that the columns are highly negatively related. The correlation can take values between [– 1,1] and a low absolute values indicates low correlation.  
![image](https://user-images.githubusercontent.com/76804249/197333650-742eee91-497f-46b8-b8f9-f65c0066d9a1.png)
- We also get the number of null values and duplicate values for a better analysis
```
The number of duplicated values in the dataset are:  68
The number of null values in the dataset are:  0
```

### **The analysis drawn from the data are**  
- There are no null values in the dataset and 68 duplicate values.
- We can see that the DERMASON is the most abundant class while BOMBAY is the least abundant class  
- BOMBAY class has the highest area and perimeter while the DERMASON class the lowest.  
- There are many outliers in the area and perimeter in the dataset  
- BOMBAY Class has high Major and Minor Axis length and Minor Axis Length.  
- ConvexArea and Area have a direct correlation. i.e., As ConvexArea increases the Area increases.  

## Part 3
- TCA helps us to analyse high dimensional data. It first measures the similarities in a high n dimensional space and then convert it to JOINT PDF to retain and reflect the probability as accurately as possible.  
- The target class is first separated to another variable.  
- We first fit the data into TSNE using fit_transform().  
- We now plot the result of the TSNE using a scatterplot  
![image](https://user-images.githubusercontent.com/76804249/197333811-b6dc8ffe-5914-4eab-b287-a79b71fc97fa.png)


- It is clear from the scatterplot that the data `cannot be separated using a straight line` as there is lot of overlap between different classes.

## Part 4
Firstly, we do Gaussian Naïve Bayes
- We first make a train test split of the data in 80:20 fashion using train_test_split()  
- We first need to import modules from the sklearn.   
- We now need to run Multinomial and Gaussian Naïve Bayes.  
- We first normalise the data using standard scalar.   
- We then fit our data in the gaussian imported from sklearn using gaussian.fit(trainX,trainY).  
- We then get predictedY output using gaussian.predict(testX).  
- We then calculate the precission score, recall score and accuracy score using precision_score,recall_score, and accuracy_score functions of sklearn.  
 
***The results obtained using Gaussian Naïve Bayes are as follows :***
```
Results of Gaussian Naive Bayes are as follows:
The model's precision is : 0.897
The model's recall is : 0.897
The model's accuracy is : 0.897
```
We now do the `Multinomial Naïve Bayes` using `MultinomialNB` of sklearn.
- We repeat the same steps as Gaussian above but we just use multinomial object from MultinomalNB().

***The results obtained using Multinomial Naive Bayes are as follows :***
```
Results of Multinomial Naive Bayes are as follows:
The model's precision is : 0.788
The model's recall is : 0.786
The model's accuracy is : 0.786
```
### Analysis of the Results
- We observe that the accuracy of Gaussian Naïve Bayes is `0.897` while that of Multinomial Naïve Bayes is `0.788`
- Hence Gaussian Naïve Bayes performs better than Multinomial Naïve Bayes for the given data

## Part 5
-	PCA is used to transform high dimensional data to lower dimensional data while trying to retain as much information as possible. It helps us to analyse more complex data with multiple features efficiently.
-	We make test and train split.
-	We then make fit_transform() on our train data to normalise the train data and transform() for the test data.
-	We now do logistic regression using logistic () from sklearn and use the data modified from PCA(n) function from sklearn for n components of PCA.

### ***The results for the different values of PCA are as follows***
-	PCA with 4 components
```
number of components are 4
variance is  0.9501988041227669
Accuracy is  0.8894601542416453
Precision is  0.8888329430312921
Recall is  0.8894601542416453
F1 is  0.8889938558614282
```

-	PCA with 6 components
 ```
 variance is  0.989198643096411
Accuracy is  0.9243481454278369
Precision is  0.9248812148305942
Recall is  0.9243481454278369
F1 is  0.9243511803525162
```

-	PCA with 8 components
 ```
 variance is  0.9993093340631198
Accuracy is  0.9280205655526992
Precision is  0.9286837346038485
Recall is  0.9280205655526992
F1 is  0.9280765809256262
```

-	PCA with 10 components
 ```
 variance is  0.9999083425762653
Accuracy is  0.9280205655526992
Precision is  0.9286837346038485
Recall is  0.9280205655526992
F1 is  0.9280765809256262
```
	
- As we observe on decreasing the PCA components from 12 to 4 the `accuracy and variance decreases` as the amount of information lost on compressing the data increases with higher compression ie. Lower PCA components.

## Part 6
-	We first do the same process as done above for the PCA
-	We do it for PCA components = 8
-	Now we plot the ROC_AUC curve using roc_curve(testY,probY) for getting coordinates corresponding to false and true positive rates,where probY is the probability estimates for the output values and roc_auc_score(testY,probY) for getting the roc_auc Score.
![image](https://user-images.githubusercontent.com/76804249/197334203-22b51e50-de27-49a3-8fa0-ab8e13a23b34.png)
- As evident from the results all the models have a roc_auc score greater than 0.9 which demonstrate excellent results and much better than random guess which will give 0.5.
- We see that the BOMBAY class have the highest roc_auc score while the SIRA class has the smallest roc_auc score.

## Part 7
-	We repeat the same steps as done in PCA for PCA components = 10
-	We get the output using logistic.predict() as done in PCA.
-	Now we get the precision_score, accuracy_score and recall_score.

## **The results are as follows :**
```
Results of Logistic regression are as follows:
The model's precision is : 0.929
The model's recall is : 0.928
The model's accuracy is : 0.928
```
```
Results of Multinomial Naive Bayes are as follows:
The model's precision is : 0.788
The model's recall is : 0.786
The model's accuracy is : 0.786
```
```
Results of Gaussian Naive Bayes are as follows:
The model's precision is : 0.897
The model's recall is : 0.897
The model's accuracy is : 0.897
```
- We find that the accuracy in the case of Logisitic Regression implementation of SKlearn the accuracy is `much better than the Gaussian Naïve Bayes`
- Accuracy for logistic is 0.931, Gaussian Naïve Bayes is 0.897 and Multinomial Naïve Bayes is 0.786.

## What's Next ❓
- We can try to implement the various other ML models like SVM etc
- We can learn more about uses from the paper supplied above and update our models accordingly.
