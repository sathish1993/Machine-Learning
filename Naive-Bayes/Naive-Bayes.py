from __future__ import division
import os
import glob
import sys
import math
import re

# Method to return list of stop words read from the stop words folder
def get_stop_words(stop_words_folder):
	stop_words = [];
	path = os.getcwd() + "/" + stop_words_folder + "/stop-words.txt";
	with open(path) as f:
		for line in f:
			stop_words.append(line.replace('\n', ''))
	return stop_words

# Helper Method to remove special characters from file
def remove_special_characters_helper(file):
	pattern = re.compile('\W+|_')
	temp_list = []
	file_object = open(file, 'r')
	for line in file_object:
		for word in line.lower().split():
			if not pattern.match(word):
				temp_list.append(word)
	return temp_list

# Method to remove special characters
def remove_special_characters(train_folder, sub_folder_name):
	source = os.getcwd() + "/" + train_folder
	clean_list = []
	path = os.path.join(source, sub_folder_name)
	file_path = path + "/*.txt"
	file_count = 0
	for file in glob.glob(file_path):
		file_count += 1
		temp_list = remove_special_characters_helper(file)
		clean_list = clean_list + temp_list
	return clean_list, file_count	

# Method to find priors
def find_priors(count1, count2):
	prior1 = count1/(count1 + count2)
	prior2 = count2/(count1 + count2)
	return prior1, prior2

# Method to find conditional probability for all the list of words
def find_conditional_probability(list_counter, file_length, unique_word_count):
	temp_dict = dict()
	for word, count in list_counter.items():
		temp_dict[word] = (count + 1)/(file_length + unique_word_count)
	return temp_dict	

# Method to find final probability to check if it is a ham or spam
def find_final_probability(word_list, probability_dict, prior):
	probability = math.log(prior, 2)
	for word in word_list:
		if probability_dict.get(word) is not None:
			probability = probability + math.log(probability_dict.get(word), 2)
	return probability

# Method to find the count of words in the file
def find_word_count(list1, list2):		
	word_counter_dict = dict()
	for word in list1:
		word_count = 0
		if word in list2:
			word_count = list2.count(word)
		word_counter_dict[word] =  word_count
	return word_counter_dict	

# Method to remove stop words from original list
def segregate_stop_words(stop_words_folder, final_ham_list, final_spam_list):
	stop_words_list =  get_stop_words(stop_words_folder)
	final_ham_list = [word for word in final_ham_list if word not in stop_words_list]
	final_spam_list = [word for word in final_spam_list if word not in stop_words_list]
	return final_ham_list, final_spam_list

# Method to train model with training data
def train_model(train_folder, with_stop_words, stop_words_folder):
	folders = os.listdir(train_folder)
	ham_list, ham_file_count = remove_special_characters(train_folder, folders[0])
	spam_list, spam_file_count = remove_special_characters(train_folder, folders[1])
	prior_ham, prior_spam = find_priors(ham_file_count, spam_file_count)
	
	final_ham_list = sorted(ham_list)
	final_spam_list = sorted(spam_list)	

	if with_stop_words == True:
		final_ham_list, final_spam_list = segregate_stop_words(stop_words_folder, final_ham_list, final_spam_list)

	temp_list = final_ham_list + final_spam_list
	temp_set = sorted(set(temp_list))
	ham_list_counter = find_word_count(temp_set, final_ham_list)
	spam_list_counter = find_word_count(temp_set, final_spam_list)	
	unique_word_count = len(temp_set)

	conditional_probablities_ham_dict = find_conditional_probability(ham_list_counter, len(final_ham_list), unique_word_count)
	conditional_probablities_spam_dict = find_conditional_probability(spam_list_counter, len(final_spam_list), unique_word_count)
	return conditional_probablities_ham_dict, conditional_probablities_spam_dict, prior_ham, prior_spam

# Method to check ham or spam based on probability values
def check_if_ham_or_spam(ham, spam):
	return 'ham' if ham > spam else 'spam'

# Method to find accuracy
def find_accuracy(value1, value2):
	return value1/value2

# Method to find matches of output with the expected output
def find_score(folder, test, ham_probability, spam_probability, prior_ham, prior_spam):
	source = os.getcwd() + "/" + test + "/" + folder
	path = os.path.join(source)
	file_path = path + "/*.txt"
	score = 1
	file_count = 0
	for file in glob.glob(file_path):
		file_count += 1
		test_list = remove_special_characters_helper(file)
		final_ham_probability = find_final_probability(test_list, ham_probability, prior_ham)				
		final_spam_probability = find_final_probability(test_list, spam_probability, prior_spam)
		output = check_if_ham_or_spam(final_ham_probability, final_spam_probability)
		if output == folder:
			score += 1
	return score, file_count				

# Method to test model
def test_model(test, ham_probability, spam_probability, prior_ham, prior_spam):
	folders = os.listdir(test)
	ham_score, ham_file_count = find_score(folders[0], test, ham_probability, spam_probability, prior_ham, prior_spam)
	spam_score, spam_file_count = find_score(folders[1], test, ham_probability, spam_probability, prior_ham, prior_spam)
	print "Accuracy %.2f\n" % find_accuracy((ham_score + spam_score),(ham_file_count + spam_file_count))

def main():
	stop_words_folder = sys.argv[1]
	train_folder = sys.argv[2]
	test_folder = sys.argv[3]

	with_stop_words = False
	print "Training model..."
	conditional_probablities_ham_dict, conditional_probablities_spam_dict, prior_ham, prior_spam = train_model(train_folder, with_stop_words, stop_words_folder)
	print "Testing model..."
	test_model(test_folder, conditional_probablities_ham_dict, conditional_probablities_spam_dict, prior_ham, prior_spam)
	
	with_stop_words = True
	print "Processing with stop words..."
	print "Training model..."
	conditional_probablities_ham_dict, conditional_probablities_spam_dict, prior_ham, prior_spam = train_model(train_folder, with_stop_words, stop_words_folder)
	print "Testing model..."
	test_model(test_folder, conditional_probablities_ham_dict, conditional_probablities_spam_dict, prior_ham, prior_spam)

if __name__ == '__main__':
	main()