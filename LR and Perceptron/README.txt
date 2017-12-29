Academic Honesty Statement 

I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else. I understand that if I am involved in plagiarism or cheating I will have to sign an official form that I have cheated and that this form will be stored in my official university record. I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that I will receive a grade of “F” for the course for any additional offense. 

Name: Sathish Kumar Santhanagopalan 
Date: 10/30/2017 
B-Number: B00669012
B-Mail: ssantha1@binghamton.edu

Language: Python

To run the code, type the following in the terminal

python LogisticRegression.py stopwords train test
python Perceptrons.py stopwords train test

Accuracy Increases with Stop words in Logistic Regression:
	Stop words are frequent , most sentences share a similar percentage of stop words. It bloats your index without providing any extra value.Removing stop words helps decrease the size of your index as well as the size of your query. Fewer terms is always a win with regards to performance. And since stop words are semantically empty, relevance scores are unaffected.

Working procedure:
	1. Input training files are stemmed using stem_train.py. This file generates files with list of stemmed words
	2. Input test files are stemmed using stem_test.py. This file generates files with list of stemmed words
	3. These files are then checked for special characters in the main file LogisticRegression.py, algorithm is followed as specified in class slide. Same procedure is repeated for Perceptrons.py
	4. Results are available in results.txt


Accuracy Comparison:
	Naive Bayes : 0.95 0.96
	
	Number of iterations = 10, Learning rate = 0.1 lambda = 1
	Logistic Regression : 0.91 0.92

	Number of iterations = 10, Learning rate = 0.1
	Perceptrons : 0.76 0.94

	Observations:
		Naive Bayes Model is more accurate than other models.
		Logistic Regression's and Perceptron's accuracy did not increase past 0.94
		More the training better the test results in every case. If learning rate is small and iterations are fewer results are bad and vice versa.
		Learning rate has to be small and max iterations will provide better results.

