Academic Honesty Statement 

I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else. I understand that if I am involved in plagiarism or cheating I will have to sign an official form that I have cheated and that this form will be stored in my official university record. I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that I will receive a grade of “F” for the course for any additional offense. 

Name: Sathish Kumar Santhanagopalan 
Date: 10/13/2017 
B-Number: B00669012
B-Mail: ssantha1@binghamton.edu

Language: Python

To run the code, type the following in the terminal

python Naive-Bayes.py stopwords train test

Accuracy Increases with Stop words:
	 Stop words are frequent , most sentences share a similar percentage of stop words. It bloats your index without providing any extra value.Removing stop words helps decrease the size of your index as well as the size of your query. Fewer terms is always a win with regards to performance. And since stop words are semantically empty, relevance scores are unaffected.

Working procedure:

1. Input training files are stemmed using stem_train.py. This file generates files with list of stemmed words
2. Input test files are stemmed using stem_test.py. This file generates files with list of stemmed words
3. These files are then checked for special characters in the main file Naive-Bayes.py and then algorithm is followed as specified in pdf
4. Results are available in results.txt
5. Answers to point estimation questions are available in Homework2_ML.pdf

