from nltk.stem import PorterStemmer
import os
import glob


def do_stem_file(file):
	stemmed_words = []
	file_object = open(file, 'r')
	stemmer = PorterStemmer()
	for line in file_object:
		for word in line.lower().split():
			stemmed_words.append(stemmer.stem(word))
	return stemmed_words


def txt_insert(test_path, listx, filex):
	path = test_path + "/" + str(filex)
	print path
	file = open(path, 'w')
	for item in listx:
		file.write("%s\n" % item)	

def find_stemmedwords_list(train_folder, sub_folder_name):
	source = os.getcwd() + "/" + train_folder
	stemmed_words = []
	path = os.path.join(source, sub_folder_name)
	x = "/test_stem/" + sub_folder_name;
	test_path = source + x
	try:
		os.makedirs(test_path)
	except OSError:
		pass	
	file_path = path + "/*.txt"
	for file in glob.glob(file_path):
		head, tail = os.path.split(file)
		name, extn = os.path.splitext(tail)
		test_list = do_stem_file(file)
		txt_insert(test_path, test_list, tail)
	

find_stemmedwords_list('test', 'ham')
find_stemmedwords_list('test', 'spam')



