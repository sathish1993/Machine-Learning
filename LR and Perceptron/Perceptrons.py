from __future__ import division
import os
import glob
import sys
import math
import re
from collections import Counter

#Method to return list of stop words
def get_stop_words(stop_words_folder):
	stop_words = [];
	path = os.getcwd() + "/" + stop_words_folder + "/stop-words.txt";
	with open(path) as f:
		for line in f:
			stop_words.append(line.replace('\n', ''))
	return stop_words

#Hepler method to remove special characters
def remove_special_characters_helper(file):
	pattern = re.compile('\W+|_')
	temp_list = []
	file_object = open(file, 'r')
	for line in file_object:
		for word in line.lower().split():
			if not pattern.match(word):
				temp_list.append(word)
	return temp_list

#Method to find frequency
def find_frequency(words_in_file):
	return Counter(words_in_file)	

#Method to remove special characters from training folder
def remove_special_characters(train_folder, total_words_list, sub_folder_name, stop_list):
	file_word_frequecy_dict = dict()
	source = os.getcwd() + "/" + train_folder
	path = os.path.join(source, sub_folder_name)
	file_path = path + "/*.txt"
	for file in glob.glob(file_path):
		words_in_file = remove_special_characters_helper(file)

		if stop_list != None:
			words_in_file = [item for item in words_in_file if item not in stop_list]

		total_words_list += words_in_file
		frequency_dict = find_frequency(words_in_file)
		file_word_frequecy_dict[file] = frequency_dict	
	return file_word_frequecy_dict, total_words_list

#Method to initialize dictionary
def initialize(n_attributes):
	temp_dict = dict()
	for i in n_attributes:
		temp_dict[i] = 0
	return temp_dict

#Method to predict if input is spam or ham
def do_predict(word_freq_dict, weights_dict):
	value = 1
	for key in word_freq_dict.keys():
		if weights_dict.get(key) != None:
			value += word_freq_dict[key] * weights_dict[key]
	return 1 if value > 0 else 0

#Method to update weights
def do_weights_calculation(predicted_value, expected_value, input_dict, weights_dict, learning_rate):
	difference = expected_value - predicted_value
	for i in input_dict.keys():
		if weights_dict.get(i) != None:
			old = weights_dict[i] 
			weights_dict[i] = float(old) + (float(learning_rate) * difference * input_dict[i])
	return weights_dict

#Method to predict if input is spam or ham
def find_Prediction(freq_dict, weights_dict, learning_rate, expected_value):
	predict_dict = dict()
	for key in freq_dict.keys():
		frequency_dict = freq_dict[key]
		predicted_value = do_predict(frequency_dict, weights_dict)
		weights_dict = do_weights_calculation(predicted_value, expected_value, frequency_dict, weights_dict, learning_rate)
	return weights_dict

#Method to train model
def train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate):
	folders = os.listdir(train_folder)
	total_words_list = []
	spam = 1
	ham = 0

	if with_stop_words:
		stop_words_list = get_stop_words(stop_words_folder)
		ham_word_freq_dict, temp_total_words_list = remove_special_characters(train_folder, total_words_list, folders[0], stop_words_list)
		spam_word_freq_dict, final_total_words_list = remove_special_characters(train_folder, temp_total_words_list, folders[1], stop_words_list)
	else:
		ham_word_freq_dict, temp_total_words_list = remove_special_characters(train_folder, total_words_list, folders[0], None)
		spam_word_freq_dict, final_total_words_list = remove_special_characters(train_folder, temp_total_words_list, folders[1], None)

	n_attributes = sorted(set(final_total_words_list))
	weights_dict = initialize(n_attributes)

	for i in range(iterations):
		weights_dict = find_Prediction(ham_word_freq_dict, weights_dict, learning_rate, ham)
		weights_dict = find_Prediction(spam_word_freq_dict, weights_dict, learning_rate, spam)
	return weights_dict

#Method to apply model to the given input
def apply_model(freq_dict, weights_dict, expected_value, count):
	for key in freq_dict.keys():
		temp_dict = freq_dict[key]
		prediction = do_predict(temp_dict, weights_dict)
		if prediction == expected_value:
			count = count + 1
	return count		

#Method to test model
def test_model(test_folder, total_weights_dict):
	folders = os.listdir(test_folder)
	total_words_list = []
	spam = 1
	ham = 0
	count = 0
	ham_word_freq_dict, temp_total_words_list = remove_special_characters(test_folder, total_words_list, folders[0], None)
	spam_word_freq_dict, final_weights_dict = remove_special_characters(test_folder, total_words_list, folders[1], None)
	temp_count = apply_model(spam_word_freq_dict, total_weights_dict, spam, count)
	final_count = apply_model(ham_word_freq_dict, total_weights_dict, ham, temp_count)
	accuracy = final_count/ (len(ham_word_freq_dict) + len(spam_word_freq_dict))
	print "Accuracy %.2f" % accuracy

def main():
	stop_words_folder = sys.argv[1]
	train_folder = sys.argv[2]
	test_folder = sys.argv[3]

	print "1. Iterations = 10, Learning rate = 0.1"
	iterations = 10
	learning_rate = 0.1
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n2. Iterations = 3, Learning rate = 0.2"
	iterations = 3
	learning_rate = 0.2
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n3. Iterations = 5, Learning rate = 1"
	iterations = 5
	learning_rate = 1
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n4. Iterations = 15, Learning rate = 5"
	iterations = 15
	learning_rate = 5
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n5. Iterations = 1, Learning rate = 10"
	iterations = 1
	learning_rate = 10
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n6. Iterations = 100, Learning rate = 0.001"
	iterations = 100
	learning_rate = 0.001
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n7. Iterations = 50, Learning rate = 0.1"
	iterations = 50
	learning_rate = 0.1
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n8. Iterations = 25, Learning rate = 0.3"
	iterations = 25
	learning_rate = 0.3
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n9. Iterations = 12, Learning rate = 0.15"
	iterations = 12
	learning_rate = 0.15
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n10. Iterations = 10, Learning rate = 0.9"
	iterations = 10
	learning_rate = 0.9
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n11. Iterations = 75, Learning rate = 0.9"
	iterations = 75
	learning_rate = 0.9
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n12. Iterations = 0, Learning rate = 10"
	iterations = 0
	learning_rate = 10
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n13. Iterations = 300, Learning rate = 0.09"
	iterations = 300
	learning_rate = 0.09
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n14. Iterations = 150, Learning rate = 0.2"
	iterations = 150
	learning_rate = 0.2
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n15. Iterations = 1, Learning rate = 30"
	iterations = 1
	learning_rate = 30
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n16. Iterations = 15, Learning rate = 30"
	iterations = 15
	learning_rate = 30
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n17. Iterations = 100, Learning rate = 0.0004"
	iterations = 100
	learning_rate = 0.004
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n18. Iterations = 20, Learning rate = 0.5"
	iterations = 20
	learning_rate = 0.5
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n19. Iterations = 200, Learning rate = 0.05"
	iterations = 200
	learning_rate = 0.05
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

	print "\n20. Iterations = 10, Learning rate = 0.15"
	iterations = 10
	learning_rate = 0.15
	print "Without Stop words"
	with_stop_words = False
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)
	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate)
	test_model(test_folder, total_weights_dict)

if __name__ == '__main__':
		main()	