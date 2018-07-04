import torch
from torch.autograd import Variable

class do_nothing(torch.nn.Module):
		def __init__(self):
			super(do_nothing, self).__init__()
		def forward(self, x):
			return x

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
    assert h == 1, "the height of out must be 1"
    features = features.permute(3, 0, 2, 1)
    features = features.squeeze(2)
    return features
