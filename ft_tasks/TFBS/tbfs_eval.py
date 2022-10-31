# evaluation and test files
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score, \
	f1_score, accuracy_score, confusion_matrix, matthews_corrcoef, average_precision_score

def evaluate(net, data_loader, device, tag="DEV",  verbose=True):

	pred_list, gold_list, pred_list_score = [], [], []

	for x, y, category in data_loader:
		x = torch.tensor(x, dtype = torch.float32).transpose(1,2)

		outputs = net(x.to(device))
		pred_list.extend(torch.argmax(outputs, dim=1).to("cpu").detach().numpy())
		pred_list_score.extend(F.softmax(outputs, dim=1).to("cpu").detach().numpy()[:,1]) 
		gold_list.extend(y)

	auc_score = roc_auc_score(gold_list, pred_list_score)
	tn, fp, fn, tp = confusion_matrix(gold_list, pred_list).ravel()  
	precision, recall = tp/(tp+fp), tp/(tp+fn)
	sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)

	acc = (tp + tn) / (tp+tn+fp+fn)
	f1  = 2*tp / (2*tp + fp + fn)
	mcc = matthews_corrcoef(gold_list, pred_list)
	auprc = average_precision_score(gold_list, pred_list_score)
	
	if verbose:
		print(" |-["+tag+" EVAL]: AUC=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, MCC=%.4f, AUPRC=%.4f" %(auc_score, precision, recall, f1, mcc, auprc))

	return mcc, auc_score, auprc, f1

def evaluate_cache(net, cached_data, device, tag="DEV",  verbose=True):

	pred_list, gold_list, pred_list_score = [], [], []

	X, Y, Cats = cached_data[0], cached_data[1], cached_data[2]

	for i in range(len(X)):
		x, y, category = X[i], Y[i], Cats[i]

	#for x, y, category in data_loader:
		x = torch.tensor(x, dtype = torch.float32).transpose(1,2)

		outputs = net(x.to(device))
		pred_list.extend(torch.argmax(outputs, dim=1).to("cpu").detach().numpy())
		pred_list_score.extend(F.softmax(outputs, dim=1).to("cpu").detach().numpy()[:,1]) 
		gold_list.extend(y)

	auc_score = roc_auc_score(gold_list, pred_list_score)
	tn, fp, fn, tp = confusion_matrix(gold_list, pred_list).ravel()  
	precision, recall = tp/(tp+fp), tp/(tp+fn)
	sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)

	acc = (tp + tn) / (tp+tn+fp+fn)
	f1  = 2*tp / (2*tp + fp + fn)
	mcc = matthews_corrcoef(gold_list, pred_list)
	auprc = average_precision_score(gold_list, pred_list_score)
	
	if verbose:
		print(" |-["+tag+" EVAL]: AUC=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, MCC=%.4f, AUPRC=%.4f" %(auc_score, precision, recall, f1, mcc, auprc))

	return mcc, auc_score, auprc, f1

