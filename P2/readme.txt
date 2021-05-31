Santosh Bhandari
1001387116

I selected  'MP','FG%', '3P%', '2P%', 'FT%', 'ORB', 'DRB', 
'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G' as a feat
users to do classfications. 
I have used k-nearest neighbor to classify the data
The accuracy was optimal when I used k as 5. I tried 
using 3 and 7 but it didn't give as good result as of 5.

Cross validation matrix gives how many data were corectly
evaluated by using the models. And which position gives the
erros. In my cross validation matrix power forward and center 
forward were close and model had misclassified power forward as
center forward. 

Accuracy of cross validation is printed in the array and average
accuracy is printed after that. 