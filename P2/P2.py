import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# "Position (pos)" is the class attribute we are predicting. 
class_column = 'Pos'

#The dataset contains attributes such as player name and team name. 
#We know that they are not useful for classification and thus do not 
#include them as features. 
#Included only the one that has the impact on deciding where the player 
#pay
feature_columns = [  'MP','FG%', '3P%', '2P%', 'FT%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']

#Pandas DataFrame allows you to select columns. 
#We use column selection to split the data into features and class. 
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

#Spliting train features and test features
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25,random_state=0)

training_accuracy = []
test_accuracy = []

#Classifying using k nearest neighbor and selecting 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5,weights = 'uniform', metric='minkowski', p=1)
knn.fit(train_feature, train_class)
prediction = knn.predict(test_feature)

# print("Test set predictions:\n{}".format(prediction))
print("Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))

train_class_df = pd.DataFrame(train_class,columns=[class_column])     
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
train_data_df.to_csv('train_data.csv', index=False)

temp_df = pd.DataFrame(test_class,columns=[class_column])
temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)
test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)
test_data_df.to_csv('test_data.csv', index=False)

prediction = knn.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

scores = cross_val_score(knn, nba_feature, nba_class, cv=10)

print("Cross-validation scores accuracy: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))