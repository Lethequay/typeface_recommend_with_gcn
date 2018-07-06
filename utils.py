import torch
from torch.autograd import Variable

def sort_sequence(data, len_data):

	_, idx_sort = torch.sort(len_data, dim=0, descending=True)
	_, idx_unsort = torch.sort(idx_sort, dim=0)

	sorted_data = data.index_select(0, idx_sort)
	sorted_len = len_data.index_select(0, idx_sort)

	return sorted_data, sorted_len.data.cpu().numpy(), idx_unsort

def unsort_sequence(data, idx_unsort):
	unsorted_data = data.index_select(0, idx_unsort)
	return unsorted_data

def features_to_sequence(features):
	b, c, h, w = features.size()
	features = features.view(b,-1,1,w)

	features = features.permute(0, 3, 1, 2)
	features = features.squeeze(3)
	return features
