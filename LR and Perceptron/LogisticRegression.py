from __future__ import division
import os
import glob
import sys
import math
import re
from collections import Counter

#Method to return stop words list
def get_stop_words(stop_words_folder):
	stop_words = [];
	path = os.getcwd() + "/" + stop_words_folder + "/stop-words.txt";
	with open(path) as f:
		for line in f:
			stop_words.append(line.replace('\n', ''))
	return stop_words

#Helper method to return list after returning special characters
def remove_special_characters_helper(file):
	pattern = re.compile('\W+|_')
	temp_list = []
	file_object = open(file, 'r')
	for line in file_object:
		for word in line.lower().split():
			if not pattern.match(word):
				temp_list.append(word)
	return temp_list

#Method to find frequency of the words in the file
def find_frequency(words_in_file):
	return Counter(words_in_file)	

#Method to remove special characters from training folders
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

#Method to initialize a dictionary
def initialize(n_attributes):
	temp_dict = dict()
	for i in n_attributes:
		temp_dict[i] = 0
	return temp_dict

#Helper method to predict if ham or spam
def do_predict(word_freq_dict, weights_dict):
	value = 1
	for key in word_freq_dict.keys():
		value += word_freq_dict[key] * weights_dict[key]
	return 1 if value > 0 else 0

#Method to predict if ham or spam
def find_Prediction(freq_dict, weights_dict):
	predict_dict = dict()
	for key in freq_dict.keys():
		frequency_dict = freq_dict[key]
		predicted_value = do_predict(frequency_dict, weights_dict)
		predict_dict[key] = predicted_value
	return predict_dict	

#Method to do weight calcuations
def do_dw_calculations(ham_word_freq_dict, spam_word_freq_dict, predict_dict_ham, predict_dict_spam, n_attributes):
	dw_arrays_dict = initialize(n_attributes)	
	spam = 1 
	ham = 0
	for i in dw_arrays_dict.keys():
		for j in spam_word_freq_dict.keys():
			temp_dict = spam_word_freq_dict[j]
			frequency = temp_dict[i]
			temp = dw_arrays_dict[i]
			dw_arrays_dict[i] = temp + frequency * (spam - predict_dict_spam[j])
		for j in ham_word_freq_dict.keys():
			temp_dict = ham_word_freq_dict[j]
			frequency = temp_dict[i]
			temp = dw_arrays_dict[i]
			dw_arrays_dict[i] = temp + frequency * (ham - predict_dict_ham[j])
	return dw_arrays_dict

#Method to update weights
def do_weights_calculation(dw_arrays_dict, weights_dict, learning_rate, lambda_value):
	for key in weights_dict.keys():
		val = weights_dict[key] + float(learning_rate) * (float(dw_arrays_dict[key]) - (float(lambda_value) * float(weights_dict[key])))
		weights_dict[key] = val
	return weights_dict	

#Method to train model
def train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value):
	folders = os.listdir(train_folder)
	total_words_list = []

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
		predict_dict_ham = find_Prediction(ham_word_freq_dict, weights_dict)
		predict_dict_spam = find_Prediction(spam_word_freq_dict, weights_dict)
		dw_arrays_dict = do_dw_calculations(ham_word_freq_dict, spam_word_freq_dict, predict_dict_ham, predict_dict_spam, n_attributes)
		final_weights_dict = do_weights_calculation(dw_arrays_dict, weights_dict, learning_rate, lambda_value)
	return final_weights_dict

#Method to predict if the test input is ham or spam
def test_predict(temp_dict, weights_dict):
	value = 0
	for temp_key in temp_dict.keys():
		if temp_key in weights_dict.keys():
			value += temp_dict[temp_key] * weights_dict[temp_key] 
	return 1 if value > 0 else 0

#Helper method to test the model
def apply_model(freq_dict, weights_dict, expected_value, count):
	for key in freq_dict.keys():
		temp_dict = freq_dict[key]
		prediction = test_predict(temp_dict, weights_dict)
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
	iterations = 10
	learning_rate = 0.1

	print "Number of iterations = 10, Learning rate = 0.1"
	with_stop_words = False	
	print "lambda = 1"
	print "Without Stop words"
	lambda_value = 1
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

	print "\nlambda = 10"
	print "Without Stop words"
	with_stop_words = False
	lambda_value = 10
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

	print "\nlambda = 5"
	print "Without Stop words"
	with_stop_words = False
	lambda_value = 5
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

	print "\nlambda = 0.5"
	print "Without Stop words"
	with_stop_words = False
	lambda_value = 0.5
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

	print "\nlambda = 0.01"
	print "Without Stop words"
	with_stop_words = False
	lambda_value = 0.01
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

	print "With Stop words"
	with_stop_words = True
	total_weights_dict = train_model(train_folder, with_stop_words, stop_words_folder, iterations, learning_rate, lambda_value)
	test_model(test_folder, total_weights_dict)

if __name__ == '__main__':
		main()	