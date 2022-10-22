### Dry Bean Class prediction using Machine Learning Techniques.
#### Motivation to solve the Problem
Dry Beans are one of the most consumed edible legumes in the world. Seed quality effects the crop production. It is hence essential to classify dry bean seeds for optimal production and max revenue as well as for sustainable farming.

### Dataset and Publications Used
- The data is obtained from the following   [`dataset`](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset).
- The motivation for the problem is sourced form the following [ `paper`](https://www.sciencedirect.com/science/article/abs/pii/S0168169919311573?via=ihub). This paper helped me to understand the underlying the scientific basis for this problem.

### Goals of the Project
1. Data Preprocessing
2. Performing EDA on the given Dataset to analyse the data better.
3. Using `TSNE` to reduce the dimensions of the data and plot the scatter plot. This will help us to comment on the seperability of the data.
4. Predicting the class using Sklearn's Gaussian and Multinomial Naive Bayes Model.
5. Using PCA for reducing the dimensions of the data and analyse the performance.
6. Plot ROC and AUC for the data using logistic regression of Sklearn.
7. Preict the class using LogisticRegression of Sklearn and compare performance betweeen two Naive Bayes Models.

## Methodology and Results Obtained.
### Part 1.
- Since we are making use of Libraries here, we need to import all the relevant libraries like numpy, matlplotlib, seaborn and pandas.
We then make a pandas dataframe from the given CSV Dataset file.
We then make a histogram and line plot according to the ‘CLASS” column of the given dataset.  
![image](https://user-images.githubusercontent.com/76804249/197333397-c78b59cc-c9e5-4666-9570-adf448b82594.png)

- We now plot the histograms of all other classes with respect to features
![image](https://user-images.githubusercontent.com/76804249/197333427-d43d0c84-4ce2-4dd1-8f6f-d51d0b20399e.png)

### **Analysis from the Data**
- Most Number of beans have a major axis length of around 250 and minor axis length of around 190.
- Area of around 42000 is the most common area
-	Most number of beans belong to the class DERMASON and the least number of beans to the class BOMBAY

### Part 2
- Here we need to perform and Explanatory Data Analysis.
We use df.info() to get the details about the dataset like number of columns, value types etc and df.describe() to get the details like count, average, maximum and minimum etc for the all the columns. We also use df.shape() to get the number of rows which is 13611 and number of columns which are 17 for our dataset.
Now first we plot the histogram for each column. This helps us better visualise the frequency of values present in a column.  
![image](https://user-images.githubusercontent.com/76804249/197333554-fdbe2ee9-983c-4497-b11f-9bd4d630898b.png)

- Now we plot a scatterplot between Area and Perimeter which better helps understand the relationship between the two and what area and perimeter is common in a class.
We observe that area is directly related to the perimeter and that the BOMBAY class tends to have the highest area and perimeter.  
![image](https://user-images.githubusercontent.com/76804249/197333587-22dbab8d-bf16-4fea-a63a-b56dd64ba369.png)

-	We now make the Boxplots for Area and perimeter to check for the outliers in the data. We find there are a lot of outliers in the data.  
![image](https://user-images.githubusercontent.com/76804249/197333616-6aea0422-dc67-44c6-a317-56e3bf7a5370.png)

-	We now plot a pairplot to given the pairwise correlation between of the columns using 
sns.pairplot() function. A high positive correlation means that the columns are highly positively related and a high negative correlation means that the columns are highly negatively related. The correlation can take values between [– 1,1] and a low absolute values indicates low correlation.  
![image](https://user-images.githubusercontent.com/76804249/197333650-742eee91-497f-46b8-b8f9-f65c0066d9a1.png)


**The analysis drawn from the data are**  
•	There are no null values in the dataset.
•	We can see that the DERMASON is the most abundant class while BOMBAY is the least abundant class  
•	BOMBAY class has the highest area and perimeter while the DERMASON class the lowest.  
•	There are many outliers in the area and perimeter in the dataset  
•	BOMBAY Class has high Major and Minor Axis length and Minor Axis Length.  
•	ConvexArea and Area have a direct correlation. i.e., As ConvexArea increases the Area increases.  

### Part 3
- TCA helps us to analyse high dimensional data. It first measures the similarities in a high n dimensional space and then convert it to JOINT PDF to retain and reflect the probability as accurately as possible.  
- The target class is first separated to another variable.  
- We first fit the data into TSNE using fit_transform().  
- We now plot the result of the TSNE using a scatterplot  
![image](https://user-images.githubusercontent.com/76804249/197333811-b6dc8ffe-5914-4eab-b287-a79b71fc97fa.png)


- It is clear from the scatterplot that the data `cannot be separated using a straight line` as there is lot of overlap between different classes.

