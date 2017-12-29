from svmutil import *
y, x = svm_read_problem('training.new')
m = svm_train(y, x, '-t 0')
b, a = svm_read_problem('validation.new')
p_label, p_acc, p_val = svm_predict(b, a, m)

y, x = svm_read_problem('training.new')
m = svm_train(y, x, '-t 1')
b, a = svm_read_problem('validation.new')
p_label, p_acc, p_val = svm_predict(b, a, m)

y, x = svm_read_problem('training.new')
m = svm_train(y, x, '-t 2')
b, a = svm_read_problem('validation.new')
p_label, p_acc, p_val = svm_predict(b, a, m)


# `svm-train' Usage
# =================

# Usage: svm-train [options] training_set_file [model_file]
# options:
# -s svm_type : set type of SVM (default 0)
# 	0 -- C-SVC		(multi-class classification)
# 	1 -- nu-SVC		(multi-class classification)
# 	2 -- one-class SVM	
# 	3 -- epsilon-SVR	(regression)
# 	4 -- nu-SVR		(regression)
# -t kernel_type : set type of kernel function (default 2)
# 	0 -- linear: u'*v
# 	1 -- polynomial: (gamma*u'*v + coef0)^degree
# 	2 -- radial basis function: exp(-gamma*|u-v|^2)
# 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
# 4 -- precomputed kernel (kernel values in training_set_file)