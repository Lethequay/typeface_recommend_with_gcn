import torch
from torch.autograd import Variable

def sort_sequence(data, len_data):

	_, idx_sort = torch.sort(len_data, dim=0, descending=True)
	_, idx_unsort = torch.sort(idx_sort, dim=0)

	sorted_data = data.index_select(0, idx_sort)
	sorted_len = len_data.index_select(0, idx_sort)

	return sorted_data, sorted_len.data.cpu().numpy(), idx_unsort

def unsort_sequence(data, idx_unsort, len_data):
	idxes = zip(idx_unsort, len_data)
	unsort_data = []
	for (batch, len) in idxes:
		unsort_data.append(data[(batch, len-1)])
	unsort_data = torch.stack(unsort_data, 0)

	return unsort_data

def features_to_sequence(features):
	b, c, h, w = features.size()
	features = features.view(b,-1,1,w)

	features = features.permute(0, 3, 1, 2)
	features = features.squeeze(3)
	return features

def accuracy_at_k(pred, label, k=5):
	batch_size = len(label)
	acc_cnt = sum([l in pred[i,:k] for i, l in enumerate(label)])

	return acc_cnt/batch_size

def baccuracy_at_k(pred, label, k=30):
	label = torch.nonzero(label)
	label_size = label.size(0)
	acc_cnt = sum([j in pred[i,:k] for (i, j) in label])

	return acc_cnt/label_size
