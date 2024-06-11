import numpy as np

import torch
import torch.nn.functional as F

import higher

from models import *
from top_k import *
from SST import *
from other_attacks import flip_label

from typeguard import typechecked
from torchtyping import TensorType

import pdb

torch.autograd.set_detect_anomaly(True)



@typechecked
def margin(
		scores: TensorType["batch": ..., torch.float32, "nodes", "classes"],
		y_true: TensorType["nodes", torch.int64]
) -> TensorType["batch": ..., "nodes": ...]:
	all_nodes = torch.arange(y_true.shape[0])
	# Get the scores of the true classes.
	scores_true = scores[..., all_nodes, y_true]

	# Get the highest scores when not considering the true classes.
	scores_mod = scores.clone()
	scores_mod[..., all_nodes, y_true] = -np.inf
	scores_pred_excl_true = scores_mod.amax(dim=-1)
	return scores_true - scores_pred_excl_true


def margin_loss_soft(scores, y_true):

	scores = scores.cuda()
	y_true = y_true.cuda()

	all_nodes = torch.arange(y_true.shape[0])

	y_true_ids = torch.topk(y_true, k=1, dim=1).indices.squeeze() #[0]

	true_mask  = torch.zeros(y_true.shape).cuda()
	true_mask[all_nodes, y_true_ids] = 1.

	# Get the scores of the true classes.
	scores_true = torch.topk(torch.mul(scores, y_true), k=1, dim=1).values.squeeze()

	# Get the highest scores when not considering the true classes.
	y_neg = 1  - true_mask.clone()

	scores_pred_excl_true = torch.topk(torch.mul(scores, y_neg), k=2, dim=1).values.squeeze() #[0] torch.mul(scores, y_neg)

	return torch.sum(scores_true.view(-1, 1) - scores_pred_excl_true, dim=1)   


def k_subset_selection(soft_top_k, log_lab, BUDGET):
	"""
	k-hot vector selection from Stochastic Softmax Tricks
	"""
	_, topk_indices = torch.topk(log_lab, BUDGET)
	X = torch.zeros_like(log_lab).scatter(-1, topk_indices, 1.0)
	hard_X = X
	hard_topk = (hard_X - soft_top_k).detach() + soft_top_k
	budget_top_k = hard_topk.T
	
	return budget_top_k


def _project(budget, values, eps = 1e-7):
	r"""Project :obj:`values`:
	:math:`budget \ge \sum \Pi_{[0, 1]}(\text{values})`."""

	values1 = values.clone().detach()
	if torch.clamp(values1, 0, 1).sum() > budget:
		left = (values1 - 1).min()
		right = values1.max()
		miu = _bisection(values1, left, right, budget)
		values = values - miu
	return torch.clamp(values, min=eps, max=1 - eps)


def _bisection(edge_weights, a, b, n_pert,
				eps=1e-5, max_iter=1e3):
	"""Bisection search for projection."""
	def shift(offset: float):
		return (torch.clamp(edge_weights - offset, 0, 1).sum() - n_pert)

	miu = a
	for _ in range(int(max_iter)):
		miu = (a + b) / 2
		# Check if middle point is root
		if (shift(miu) == 0.0):
			break
		# Decide the side to repeat the steps
		if (shift(miu) * shift(a) < 0):
			b = miu
		else:
			a = miu
		if ((b - a) <= eps):
			break
	return miu


#seeding
def reset_seed(SEED=0):
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)


@torch.no_grad()
def weight_reset(m: torch.nn.Module):
	# - check if the current module has reset_parameters & if it's callabed called it on m
	reset_parameters = getattr(m, "reset_parameters", None)
	if callable(reset_parameters):
		m.reset_parameters()


def train(model, data, opt, y_train, args=None):

	opt.zero_grad()
	model.train()

	if args.model == 'CPGCN':
		out, pred_cluster = model(data.x, data.edge_index, None)
		loss_pred = F.cross_entropy(out[data.train_mask], y_train)
		loss_cluster = F.cross_entropy(pred_cluster[data.train_mask], model.clusters[data.train_mask])
		loss = loss_pred + args.lamb_c * loss_cluster
		# loss = loss_cluster
	else:
		out = model(data)[data.train_mask]
		loss = F.cross_entropy(out, y_train)

	loss.backward()
	opt.step()


def test(model, data):
	model.eval()
	out = model(data)
	val_acc = (out.argmax(1)[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
	test_acc = (out.argmax(1)[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

	return val_acc, test_acc


def meta_attack_labels(model, model_name, dataset, args, BUDGET, device, verbose=False):

	reset_seed()

	data = dataset.data

	num_classes = data.y.max() + 1

	#poison_model = MLP(dataset, args).to(device) #poison_network

	eye = torch.eye(num_classes).to(device)

	inner_opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) #args.lr #args.weight_decay

	"""
	Threat model setting where the labels for test nodes are not available.
	We use the test predictions to perform label attacks 
	"""

	if args.yhat:
		# pretrain target model
		for epoch in range(100):
			train(model, data, inner_opt, data.y[data.train_mask], args)

		if args.model == 'CPGCN':
			out, pred_cluster = model(data.x, data.edge_index, None)
			preds = F.softmax(out, dim=1).argmax(1)
		else:
			preds = F.softmax(model(data), dim=1).argmax(1)

		#reset training labels in preds
		preds[data.train_mask] = data.y[data.train_mask]

		Y = eye[preds]
		y_gt = Y.argmax(1)
	
	else:
		Y = eye[data.y]
		y_gt = Y.argmax(1)

	"""
	Setting 1: flip two most popular classes
	Setting 2: randomly flip train nodes classes to false classes
	Setting 3: flip based on target model logits
	"""

	# For binary attack get the indices of top-2 classes
	if args.binary_attack:

		if args.binary_setting == 1:
			class_a, class_b = torch.bincount(Y.argmax(1)).argsort(descending=True)[:2] 

			ab_train_ids = []
			
			for idx in range(data.train_mask.sum()):
				label = y_gt[data.train_mask][idx]
				if label == class_a or label == class_b:
					ab_train_ids.append(idx) 

			print("Total ab ids:", len(ab_train_ids))

			#construct Y_L_flipped
			class_a_ids = torch.where(y_gt[data.train_mask] == class_a)[0]
			class_b_ids = torch.where(y_gt[data.train_mask] == class_b)[0]
			
			y_gt_copy = y_gt[data.train_mask].clone()
			y_gt_copy[class_a_ids] = class_b
			y_gt_copy[class_b_ids] = class_a
			
			Y_L_flipped = torch.eye(num_classes)[y_gt_copy].cuda()		

		if args.binary_setting == 2:

			ab_train_ids = range(data.train_mask.sum())

			Y_L_flipped = []
			for train_label in Y[data.train_mask]:
				Y_L_flipped.append(flip_label(train_label.cpu().detach().numpy(), num_classes))
			
			Y_L_flipped = torch.tensor(Y_L_flipped).cuda()

		elif args.binary_setting == 3:

			ab_train_ids = range(data.train_mask.sum())

			# pretrain target model
			for epoch in range(10):
				train(model, data, inner_opt, y_gt[data.train_mask], args)

			if args.model == 'CPGCN':
				out, pred_cluster = model(data.x, data.edge_index, None)
				train_preds = F.softmax(out[data.train_mask], dim=1)
			else:
				train_preds = F.softmax(model(data)[data.train_mask], dim=1)

			Y_false = 1 - Y[data.train_mask]

			Y_L_flipped = eye[torch.argmax(train_preds * Y_false, 1)]

			# reset model weights
			model.apply(weight_reset)


	# top-k based log-lab
	top_k = SoftTopK() # SST based soft-top-k

	if args.naive_log_lab:
		# Naive log-lab
		log_lab = torch.nn.Parameter(torch.log(eye[data.y[data.train_mask]]+0.01))
		log_lab_H = torch.nn.Parameter(torch.zeros(data.train_mask.sum(), 1))
		torch.nn.init.uniform_(log_lab_H)
		# torch.nn.init.uniform_(log_lab)
	else:
		log_lab = torch.nn.Parameter(torch.zeros(1, data.false_onehot.shape[0]).to(device)) # for SIMPLE top-k
		torch.nn.init.uniform_(log_lab)

	if args.binary_attack:
		# H = torch.nn.Parameter(torch.zeros(Y_L_flipped.shape[0], 1))
		H = torch.nn.Parameter(torch.zeros(len(ab_train_ids), 1))
		torch.nn.init.uniform_(H)
		meta_opt = torch.optim.Adam([H], lr=0.1, weight_decay=1e-5)
	elif args.naive_log_lab:
		meta_opt = torch.optim.Adam([log_lab, log_lab_H], lr=10., weight_decay=1e-5) 
	else:
		meta_opt = torch.optim.Adam([log_lab], lr=0.1, weight_decay=1e-5)

	# for tracking best poisoned labels w.r.t meta test accuracy
	best_test_acc = 200
	best_poisoned_labels = None
	PATIENCE = 20
	patience_ctr = 0

	for ep in range(100): #150
		#poison_model.train()

		if args.binary_attack:

			BUDGET = min(BUDGET, H.shape[0])
			soft_top_k = top_k.apply(H.T, BUDGET, 1e-2)
			budget_top_k = k_subset_selection(soft_top_k, H.T, BUDGET).cuda()


			zeros_vec = torch.zeros(Y_L_flipped.shape[0], 1).cuda()
			zeros_vec[ab_train_ids] = budget_top_k
			poisoned_ = (Y_L_flipped * zeros_vec)
			correct_ = Y[data.train_mask] * (1 - zeros_vec)

			poisoned_labels = poisoned_ + correct_

		else:

			if args.topk_type == 'soft' and args.sampling and not args.naive_log_lab:
				soft_top_k = top_k.apply(log_lab, 2*BUDGET, 1e-2) # SST based soft-top-k
				budget_top_k = soft_top_k.T

			elif args.topk_type == 'hard' and not args.naive_log_lab:
				# hard_top_k from SST
				soft_top_k = top_k.apply(log_lab, BUDGET, 1e-2) # SST based soft-top-k
				budget_top_k = k_subset_selection(soft_top_k, log_lab, BUDGET)

			elif args.topk_type == 'pgd' and not args.naive_log_lab:
				budget_top_k = _project(BUDGET, log_lab).T


			if args.naive_log_lab:

				one_hot_loglab = F.gumbel_softmax(log_lab, tau=1000, hard=True, dim=1)

				# select which node labels to pertrurb
				soft_top_k = top_k.apply(log_lab_H.T, BUDGET, 1e-2)
				budget_top_k = k_subset_selection(soft_top_k, log_lab_H.T, BUDGET).cuda()

				poisoned_labels = ((1-budget_top_k) * data.y_train_onehot.cuda() + budget_top_k*one_hot_loglab).cuda()

			else:
				H = torch.mul(data.false_onehot.cuda(), budget_top_k)
				H_reshaped = torch.reshape(H, (data.y_train_onehot.shape[0], data.y_train_onehot.shape[1], data.y_train_onehot.shape[1]))
				H_final = torch.sum(H_reshaped, dim=1)
				
				P_reshaped = torch.reshape(budget_top_k, (data.y_train_onehot.shape[0], data.y_train_onehot.shape[1], 1)).sum(dim=1)

				poisoned_labels = ((1-P_reshaped) * data.y_train_onehot.cuda() + H_final).cuda()



		with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
			fmodel.train()

			for epoch in range(15):
				if model_name == 'CPGCN':

					pred, pred_cluster = fmodel(data.x, data.edge_index, None) 

					loss_pred = F.cross_entropy(pred[data.train_mask], poisoned_labels)
					loss_cluster = F.cross_entropy(pred_cluster[data.train_mask], fmodel.clusters[data.train_mask])

					lam = args.lamb_c # from LafAK paper
					loss = loss_pred + lam * loss_cluster
			
				else:

					out = fmodel(data)

					if args.naive_log_lab:
						loss = F.cross_entropy(out[data.train_mask], poisoned_labels)

					else:
						# soft margin loss
						# loss = -margin_loss_soft(out[data.train_mask], poisoned_labels).tanh().mean()

						loss = F.cross_entropy(out[data.train_mask], poisoned_labels) #F.softmax(poisoned_labels, dim=1)

				diffopt.step(loss)


			fmodel.eval()
			if model_name == 'CPGCN':
				meta_out, _ = fmodel(data.x, data.edge_index, None)
			else:
				meta_out = fmodel(data)

			# train_acc = (meta_out.argmax(1)[data.train_mask] == log_lab.argmax(1)).sum() / data.train_mask.sum() 
			train_acc = (meta_out.argmax(1)[data.train_mask] == poisoned_labels.argmax(1)).sum() / data.train_mask.sum()
			acc = (meta_out.argmax(1)[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

			# n_poisoned = (log_lab.argmax(1) != data.y[data.train_mask]).sum() 
			n_poisoned = (poisoned_labels.argmax(1) != data.y[data.train_mask]).sum()


			# early stopping based on meta-test acc
			if (acc < best_test_acc):
				best_test_acc = acc
				best_poisoned_labels = poisoned_labels.type(torch.float32).detach() #log_lab.detach()
				patience_ctr = 0 
			else:
				patience_ctr += 1

			if patience_ctr >= PATIENCE:
				break


			# convert meta out to binary using top-k
			meta_test_out = meta_out[data.test_mask]

			# 0-1 gumbel loss
			meta_out_bin = F.gumbel_softmax(meta_test_out, tau=100, hard=True, dim=1)
			meta_loss = (Y[data.test_mask] * meta_out_bin).sum()/len(meta_out_bin)


			if verbose:
				# print('acc', acc.cpu().numpy(), 'n_poisoned', n_poisoned.cpu().numpy(), 'diff', diff.cpu().numpy())

				print("epoch: {:4d} \t Meta-Train Acc: {:.2f} \t Meta-Test Acc: {:.2f} \t N-poisoned: {:4d} \t patience ctr: {:2d} \t MetaLoss: {:.4f}".format(ep, \
																						train_acc.cpu().numpy(), acc.cpu().numpy()*100, \
																						n_poisoned.cpu().numpy(), patience_ctr,
																						meta_loss.detach().cpu().numpy())) #diff.cpu().numpy()

			meta_opt.zero_grad()
			meta_loss.backward()
			meta_opt.step()

	if verbose:
		print("Best Test Acc (attacked): {:.2f}".format(best_test_acc * 100))
	
	#poison_model.apply(weight_reset)
	torch.cuda.empty_cache()

	# return log_lab.detach()
	return best_poisoned_labels



