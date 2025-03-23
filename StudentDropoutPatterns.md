### Understanding Student Dropout Patterns Through Performance and Learning Behaviors
### by Sharon Viers
### 
#### Data set: https://www.kaggle.com/datasets/adilshamim8/personalized-learning-and-adaptive-education-dataset by Adil Shamim  
#### Introduction: 
#### This dataset focuses on personalized learning and adaptive education, capturing detailed student interaction data from online learning platforms. It includes a variety of features related to student demographics, engagement, and academic performance. Each student is identified by a unique ID, with additional information such as age, gender, and education level. The dataset tracks activity across different online courses, logging time spent watching videos, quiz attempts and scores, participation in forum discussions, and assignment completion rates. It also measures engagement levels and final exam scores, alongside learning style preferences and feedback ratings. A key feature is the dropout likelihood indicator, which marks whether a student is at risk of discontinuing the course. 

#### Identify a question or questions that you would like to explore in your data set.
#### Questions:
#### 1. What factors are most strongly associated with a high likelihood of student dropout?
#### 2. How does learning style impact final exam performance?
#### 3. Is there a correlation between time spent on videos and quiz or final exam performance?

#### Create at least three graphs that help answer these questions. 


```python
# Question 1. What factors are most strongly associated with a high likelihood of student dropout?
# First, find out the percentage of dropouts.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\sharo\OneDrive\Desktop\Bellevue\Data Mining\personalized_learning_dataset.csv')

# Count the values in the 'Dropout_Likelihood' column
dropout_counts = df['Dropout_Likelihood'].value_counts()

# Set up pie chart components
labels = dropout_counts.index
sizes = dropout_counts.values
colors = ['lightcoral', 'skyblue']
explode = (0.1, 0) if 'Yes' in labels else (0, 0)

# Create the pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Dropout Likelihood Distribution')
plt.axis('equal') 
plt.show()
```


    
![png](output_3_0.png)
    


#### The above pie chart shows that 19.6% percent of students drop out of school and do not graduate.


```python
# Dropouts vs Learning Style
import seaborn as sns

# Set Seaborn style for cleaner visuals
sns.set(style="whitegrid")

# Create bar chart: Learning Style vs Dropout Likelihood
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Learning_Style', hue='Dropout_Likelihood', palette='Set3')

# Customize plot
plt.title('Dropout Likelihood by Learning Style')
plt.xlabel('Learning Style')
plt.ylabel('Number of Students')
plt.legend(title='Dropout Likelihood')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```


    
![png](output_5_0.png)
    


#### The above bar chart highlights the importance of offering varied instructional methods in education to support different learning styles and reduce dropout risk. Even with the varied instructional methods, the number of students that dropped out of school seem to be almost the same for each category of learning style. This suggests that learning style alone may not be a strong predictor of whether a student will drop out.


```python
# Dropout likelihood by quiz score

# Create bins for quiz scores 
df['Quiz_Score_Bin'] = (df['Quiz_Scores'] // 5) * 5

# Calculate dropout percentages within each bin
dropout_by_bin = (
    df.groupby('Quiz_Score_Bin')['Dropout_Likelihood']
    .value_counts(normalize=True)
    .unstack()
    .fillna(0) * 100
)

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(dropout_by_bin.index, dropout_by_bin['Yes'], marker='o', label='Dropout %', color='red')
plt.plot(dropout_by_bin.index, dropout_by_bin['No'], marker='o', label='Stayed %', color='green')

# Plot the chart
plt.title('Dropout Likelihood by Quiz Score')
plt.xlabel('Quiz Score (%)')
plt.ylabel('Percentage of Students')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_7_0.png)
    


#### The above line graph displaying dropout likelihood by quiz score range reveals a clear trend: as quiz scores increase, the percentage of students likely to drop out decreases. Students with the lowest quiz scores are at the highest risk of dropping out, while those with higher scores tend to stay in the course. The graph also shows that students who do not drop out are more commonly found in the higher score ranges, indicating that strong quiz performance is associated with higher retention. This suggests that quiz scores can serve as a useful indicator for identifying students who may need additional support.


```python
# Question 2. How does learning style impact final exam performance?

# Final exam performance by learning style
# Map learning styles to numeric values for plotting
learning_style_map = {style: i for i, style in enumerate(df['Learning_Style'].unique())}
df['Learning_Style_Num'] = df['Learning_Style'].map(learning_style_map)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(
    df['Learning_Style_Num'],
    df['Final_Exam_Score'],
    alpha=0.5,
    c='mediumseagreen',
    edgecolors='w',
    s=60
)

# Adjust x-axis labels
plt.xticks(ticks=list(learning_style_map.values()), labels=list(learning_style_map.keys()))

# Plot the chart
plt.title('Final Exam Performance by Learning Style')
plt.xlabel('Learning Style')
plt.ylabel('Final Exam Score (%)')
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_9_0.png)
    


#### From the scatter plot we observe that final exam scores are widely spread across all learning styles. No single learning style consistently outperforms the others, and each group includes students with both high and low scores. This suggests that a student’s preferred learning style, whether visual, auditory, reading/writing, or kinesthetic, does not have a strong impact on final exam performance. 


```python
# Question 3. Is there a correlation between time spent on videos and quiz or final exam performance?

# Correlation between time spent on videos and performance
correlation_data = {
    'Quiz Scores': df['Time_Spent_on_Videos'].corr(df['Quiz_Scores']),
    'Final Exam Scores': df['Time_Spent_on_Videos'].corr(df['Final_Exam_Score'])
}

# Create bar chart
plt.figure(figsize=(8, 6))
plt.bar(correlation_data.keys(), correlation_data.values(), color=['royalblue', 'seagreen'])

# Customize the chart
plt.title('Correlation Between Time Spent on Videos and Performance')
plt.ylabel('Correlation Coefficient')
plt.ylim(-0.05, 0.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```


    
![png](output_11_0.png)
    


#### The bar chart showing the correlation between Time Spent on Videos and Performance reveals that there is almost no relationship between how much time students spend watching course videos and how well they perform on quizzes or the final exam. Both correlation values were very close to zero, one slightly positive, the other slightly negative, indicating that video watching alone doesn't predict academic success. This suggests that while videos may support learning, simply spending more time on them doesn't guarantee better outcomes.

#### Conclusion

#### The analysis of the Personalized Learning & Adaptive Education dataset through selected visualizations reveals important patterns related to student performance and dropout likelihood. The Dropout Likelihood Distribution chart highlights that a significant portion of students are at risk of dropping out, emphasizing the need to understand contributing factors.

#### One of the most notable findings is that quiz performance has a strong relationship with dropout risk. The chart showing Dropout Likelihood by Quiz Score clearly demonstrates that students with lower quiz scores are more likely to drop out, suggesting that quiz results can serve as an early indicator for academic disengagement.

#### In contrast, learning style appears to have little influence on either dropout likelihood or final exam performance. Charts comparing Dropout Likelihood by Learning Style and Final Exam Performance by Learning Style show similar distributions across all learning preferences, indicating that no single style leads to significantly better or worse outcomes.

#### Finally, the chart showing the Correlation Between Time Spent on Videos and Performance reveals that time spent watching videos has almost no correlation with quiz or final exam scores. This suggests that passive engagement with content does not strongly impact student success.

#### Overall, the analysis points to academic performance, particularly quiz scores, as a more meaningful predictor of student success than learning style or time spent on videos. These insights can help guide early interventions and support strategies in online learning environments.
