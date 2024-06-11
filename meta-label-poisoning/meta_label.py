import os
import torch
from tqdm import tqdm
import argparse
import copy
import pickle

import torch
import torch.nn.functional as F

import optuna
from optuna.samplers import TPESampler, RandomSampler

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedShuffleSplit

from models import *
from attack import *
from other_attacks import *
from utils import *
from margin_attack import margin_attack_labels
from easydict import EasyDict
import wandb

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_best_trial(study):
	# Picks the trial with Best Val Accuracy
	# On conflict, picks the trial with best Test Accuracy
	trials = study.trials.copy()
	val_accs = np.array([t.user_attrs['Val Accuracy'] for t in study.trials])
	test_accs = np.array([t.user_attrs['Test Accuracy'] for t in study.trials])
	best_val = np.max(val_accs)
	best_val_indices = np.array(np.argwhere(val_accs == best_val).reshape(-1))
	best_index = best_val_indices[np.argmax(test_accs[best_val_indices]).reshape((-1))[0]]
	return trials[best_index]


def train_eval(model, y_train, args, verbose=False, emb=False):

	def train(model, data, opt, y_train):
		model.train()

		if args.model == 'CPGCN':
			out, pred_cluster = model(data.x, data.edge_index, None)
			loss_pred = F.cross_entropy(out[data.train_mask], y_train)
			loss_cluster = F.cross_entropy(pred_cluster[data.train_mask], model.clusters[data.train_mask])
			loss = loss_pred + args.lamb_c * loss_cluster
		else:
			out = model(data)[data.train_mask]
			loss = F.cross_entropy(out, y_train)

		opt.zero_grad()
		loss.backward()
		opt.step()

		del out

	def test(model, data):

		model.eval()

		if args.model == 'CPGCN':
			out, pred_cluster = model(data.x, data.edge_index, None)
		else:
			out = model(data)

		# using poisoned labels for computing validation accuracy
		val_acc = (out.argmax(1)[data.val_mask] == y_poisoned[data.val_mask]).sum() / data.val_mask.sum() 
		test_acc = (out.argmax(1)[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

		return val_acc, test_acc, out


	opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model.apply(weight_reset)

	best_val_acc = test_acc = 0
	best_emb = None
	val_acc_history = []
	predictions = []

	for epoch in range(args.epochs):
		train(model, data, opt, y_train)

		val_acc, tmp_test_acc, out = test(model, data)

		if verbose:
			print("epoch: {:4d} \t Val Acc: {:.2f} \t Test Acc: {:.2f}".format(epoch, val_acc*100, tmp_test_acc*100))

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			test_acc = tmp_test_acc
			best_emb = out

		if epoch >= 0:
			val_acc_history.append(val_acc)
			if args.early_stopping > 0 and epoch > args.early_stopping:
				tmp = torch.tensor(
					val_acc_history[-(args.early_stopping + 1):-1])
				if val_acc < tmp.mean().item():
					# print('EARLY STOPPING!')
					break

	if emb:
		torch.save(best_emb, 'embedding')

	return best_val_acc, test_acc


def objective(trial):

	if args.hyp_param_tuning:
		args.lr = trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.05, 0.08]) 
		args.weight_decay = trial.suggest_categorical('weight_decay', [0.0, 0.005, 0.0005, 0.00005]) 
		args.dropout = trial.suggest_categorical('dropout', [0.3, 0.5, 0.7]) 

		if args.model == 'APPNP':
			args.alpha = trial.suggest_categorical('alpha', [0.1, 0.3, 0.5, 0.8])

		elif args.model == 'CPGCN':
			args.lamb_c = trial.suggest_categorical('alpha', [0.5, 1.0])

	if args.model == 'CPGCN':
		OPTUNA_model = copy.deepcopy(model)
		OPTUNA_model.initialize()
	else:
		OPTUNA_model = network(dataset, args).to(device) 

	#evaluate on n CV folds
	val_acc_log, test_acc_log = [], []
	for idx in range(len(CV_train_val_set)):

		if args.setting == 'cv':

			train_index, val_index = CV_train_val_set[idx][0], CV_train_val_set[idx][1]
			
			# reset train_mask and val_mask
			data.train_mask = index_to_mask(dataset.train_ids[train_index], size=dataset.dataset_data['X'].shape[0])	
			data.val_mask = index_to_mask(dataset.train_ids[val_index], size=dataset.dataset_data['X'].shape[0])

		y_train_poisoned = y_poisoned[data.train_mask] 

		v_acc, t_acc = train_eval(OPTUNA_model, y_train_poisoned, args, verbose=False) #False
		
		val_acc_log.append(v_acc.detach().cpu().numpy())
		test_acc_log.append(t_acc.detach().cpu().numpy())

		if args.model == 'GCN':
			trial.set_user_attr('conv1', OPTUNA_model.conv1.lin.weight.detach().cpu().numpy())
			trial.set_user_attr('conv2', OPTUNA_model.conv2.lin.weight.detach().cpu().numpy())
		
		# trail.set_user_attr('rep', OPTUNA_model.)
		OPTUNA_model.apply(weight_reset) 
		reset_seed()

	avg_val_acc = np.mean(val_acc_log)
	avg_test_acc = np.mean(test_acc_log)

	#print("val acc: {:.2f} \t test acc: {:.2f}".format(avg_val_acc*100, avg_test_acc*100))

	trial.set_user_attr('Val Accuracy', avg_val_acc*100)
	trial.set_user_attr('Test Accuracy', avg_test_acc*100)
	trial.set_user_attr('Test Acc Log', test_acc_log)
	trial.set_user_attr('Val Acc Log', val_acc_log)

	return avg_val_acc


def train_N2V(dataset, device):

	# check if emb already exists
	if os.path.isfile('{}_n2v.pt'.format(dataset.name)):
		return torch.load('{}_n2v.pt'.format(dataset.name)).detach()

	from torch_geometric.nn import Node2Vec

	data = dataset.data.to(device)

	model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
					 context_size=10, walks_per_node=10,
					 num_negative_samples=1, p=1, q=1, sparse=True).to(device)	

	num_workers = 4

	loader = model.loader(batch_size=128, shuffle=True,
						  num_workers=num_workers)

	optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

	def train():
		model.train()
		total_loss = 0
		for pos_rw, neg_rw in loader:
			optimizer.zero_grad()
			loss = model.loss(pos_rw.to(device), neg_rw.to(device))
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		return total_loss / len(loader)

	@torch.no_grad()
	def test():
		model.eval()
		z = model()
		acc = model.test(z[data.train_mask], data.y[data.train_mask],
						 z[data.test_mask], data.y[data.test_mask],
						 max_iter=150)
		return acc, z

	print("Training Node2Vec")
	cur_acc = 0
	best_emb = None
	for epoch in range(1, 101):
		loss = train()
		acc, emb = test()
		print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

		if acc > cur_acc:
			acc = cur_acc
			best_emb = emb
			torch.save(emb, '{}_n2v.pt'.format(dataset.name))

	return best_emb.detach()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--early_stopping', type=int, default=100)
	parser.add_argument('--hidden', type=int, default=64)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--dataset', type=str, default='cora_ml')
	parser.add_argument('--epochs', type=int, default=1000)
	parser.add_argument('--model', type=str, default='GCN', \
											choices=['MLP', 'GCN', 'GAT', 'APPNP', 'GCN_JKNet', 'CPGCN'])
	
	parser.add_argument('--attack', type=str, default='meta', choices=['meta', 'margin_lp', 'MG', 
																		'lafak', 'random', 'degree', 
																		'lp', 'sgcbin', 'sgcbin3', 
																		'Rsgcbin', 'sgcmulti', 'ntk',
																		'random_bin', 'degree_bin'])

	parser.add_argument('--setting', type=str, default='cv', choices=['small_val', 'large_val', 'cv'])
	parser.add_argument('--hyp_param_tuning', action='store_true')
	parser.add_argument('--random_train_val', action='store_true')
	parser.add_argument('--defence', action='store_true')
	
	# Meta poisoning related hyper-params
	parser.add_argument('--topk_type', type=str, choices=['soft', 'hard', 'pgd'], default='hard')
	parser.add_argument('--sampling', action='store_true')
	parser.add_argument('--samples', type=int, default=1)
	parser.add_argument('--naive_log_lab', action='store_true')
	parser.add_argument('--binary_attack', action='store_true')
	parser.add_argument('--binary_setting', type=int, choices=[1, 2, 3], default=1)
	parser.add_argument('--yhat', action='store_true', help='use test-predictions instead of gt labels for attacks')

	#CPGCN hyper-param
	parser.add_argument('--lamb_c', type=float, default=1.0)

	#APPNP hyper-param
	parser.add_argument('--K', type=int, default=10)
	parser.add_argument('--alpha', type=float, default=0.3)

	#GAT hyper-param
	parser.add_argument('--heads', default=8, type=int)
	parser.add_argument('--output_heads', default=1, type=int)

	parser.add_argument('--wandb', action='store_true')
	parser.add_argument('--wandb_project', type=str, default='2023', help='wandb project name')
	parser.add_argument('--output_dir', type=str, default='./logs', help='path to save the logs')

	logs = EasyDict()

	args = parser.parse_args()
	args_temp = copy.deepcopy(args)

	logs.init_args = args

	if args.wandb:
		wandb.init(project=args.wandb_project)
		wandb.config.update(args)
		run_name = wandb.run.name
	else:
		run_name = f'{args.dataset}_{args.model}'

	if args.model == 'GCN':
		network = GCN
	elif args.model == 'MLP':
		network = MLP
	elif args.model == 'GAT':
		network = GAT
	elif args.model == 'APPNP':
		network = APPNP_net
	elif args.model == 'GCN_JKNet':
		network = GCN_JKNet
	elif args.model == 'CPGCN':
		network = CPGCN

	weights_log = {}

	for poison_ratio in [5, 10, 15, 20, 30]: 
		
		weights_log_splits = {}

		#logging
		logs[f'poison_ratio_{poison_ratio}'] = EasyDict()

		test_accs = []
		val_accs = []

		diffused_ypois_log = []
		ypois_log = []

		for split in tqdm(range(10)): 
			
			#logging
			logs[f'poison_ratio_{poison_ratio}'][f'split_{split}'] = EasyDict()
			
			args = copy.deepcopy(args_temp)
			reset_seed()

			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

			dataset = DataLoader(args.dataset, split=split, setting=args.setting, random_train_val=args.random_train_val)

			BUDGET = np.ceil(len(dataset.train_ids) * (poison_ratio/100)).astype(int)

			if args.model == 'CPGCN':

				from sklearn.cluster import KMeans

				# train N2V and get embedding
				embedding = train_N2V(dataset, device).cpu().numpy()

				kmeans = KMeans(dataset.num_classes, random_state=123)
				kmeans.fit(embedding)
				community_labels = kmeans.labels_
	
				model = CPGCN(nfeat=dataset.features.shape[1], 
							  nhid=args.hidden, 
							  nclass=dataset.num_classes, 
							  ncluster=dataset.num_classes, 
							  dropout=args.dropout, device=device).to(device) 
				
				model.clusters = torch.LongTensor(np.array(community_labels)).to(device)

			else:
				atk_model = network(dataset, args)
				model = atk_model.to(device)

			# move data to device
			data = dataset.data.to(device)

			# original labels
			if poison_ratio == 0:
				y = data.y[data.train_mask].cpu()

			elif args.attack == 'meta':

				log_lab = meta_attack_labels(model, args.model, 
											 dataset, args, 
											 BUDGET, device, 
											 verbose=False) 

				y = log_lab.argmax(1)


			elif args.attack == 'margin_lp':
				y = margin_attack_labels(model, dataset, args, BUDGET, device, verbose=False).cpu() #FIX


			elif args.attack == 'MG':

				MG_out = gradient_max(adj=sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.csr_matrix(dataset.adj))), \
								 features=dataset.features, \
								 idx_train=dataset.train_ids, \
								 labels=data.y.cpu().detach().numpy(), \
								 noise_num=BUDGET, \
								 args=args)
				y_poisoned = torch.tensor(MG_out, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()


			elif args.attack in ['sgcbin', 'sgcmulti', 'sgcbin3', 'Rsgcbin']:
				if args.setting == 'cv':
					# poisoning train + val
					if args.attack == 'sgcbin':
						if args.yhat:
							pkl_data = pkl.load(open("SGCbin_yhat_cv_labels/{}_sgcBin_yhat_cv_poisoned_labels.pkl".format(args.dataset), 'rb'))
						else:	
							pkl_data = pkl.load(open("./sbin_neurips_cv/{}_sgcBin_cv_poisoned_labels.pkl".format(args.dataset), 'rb'))
					elif args.attack =='sgcmulti':
						pkl_data = pkl.load(open("./smulti_neurips_cv/{}_sgcmulti_cv_poisoned_labels.pkl".format(args.dataset), 'rb'))
					elif args.attack == 'sgcbin3':
						pkl_data = pkl.load(open("./sbin_neurips_cv/{}_sgcBin3_cv_poisoned_labels.pkl".format(args.dataset), 'rb'))
					elif args.attack == 'Rsgcbin':
						pkl_data = pkl.load(open("./sbin_neurips_cv/{}_Relaxed_sgcBin_cv_poisoned_labels.pkl".format(args.dataset), 'rb'))



				if args.setting == 'small_val' and args.random_train_val:
					pkl_data = pkl.load(open("../SGCbin_small_val_random/{}_sgcbin_smallValRandom_poisoned_labels.pkl".format(args.dataset), 'rb'))

					# cora-ml-tiny
					pkl_data = pkl.load(open("../{}_sgcBinOpt_smallValRandom_poisoned_labels.pkl".format(args.dataset), 'rb'))

				y_poisoned = pkl_data['split_{}'.format(split)]['{}_percent_poison'.format(poison_ratio)].argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()

				if args.defence:
					# A^2 Y_poisoned
					ad = propagation_matrix(normalize_adj(sp.csr_matrix(dataset.adj.todense()[dataset.train_ids, :][:, dataset.train_ids]))).astype(np.float32) #.todense()
					m = torch.tensor(ad).to(device)
					yp_tr = torch.eye(y_poisoned.max()+1).cuda()[y_poisoned[data.train_mask]].to(device)
					y = (m@yp_tr).cpu() #propagation_matrix(A_hat) #(m@m)

				else:
					y = y_poisoned[data.train_mask].cpu()
				

			elif args.attack == 'ntk':
				if args.setting == 'cv':
					# poisoning train + val
					pkl_data = pkl.load(open("ntk_attack/{}_ntk_cv_poisoned_labels.pkl".format(args.dataset), 'rb'))

				if args.setting == 'small_val' and args.random_train_val:
					pkl_data = pkl.load(open("../{}_ntkbin_smallValRandom_poisoned_labels.pkl".format(args.dataset), 'rb'))

				y_poisoned = pkl_data['split_{}'.format(split)]['{}_percent_poison'.format(poison_ratio)].argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()


			elif args.attack == 'lafak':
				if args.setting == 'cv':
					# poisoning train + val
					pkl_data = pkl.load(open("./lafak_random_cv_labels/{}_lafak_random_cv_poisoned_labels.pkl".format(args.dataset), 'rb'))


				elif args.setting == 'small_val':
					# poisoning train
					if args.random_train_val:
						path = "../../LafAK/lafak_small_val_random_labels/{}_lafak_smallValRandom_poisoned_labels.pkl".format(args.dataset)
						pkl_data = pkl.load(open(path, 'rb'))
					else:
						pkl_data = pkl.load(open("../lafak_poisoning_train/{}_lafak_poisoned_labels.pkl".format(args.dataset), 'rb'))

				elif args.setting == 'large_val':
					pkl_data = pkl.load(open("../../final_data/lafak_largeVal/{}_lafak_largeVal_poisoned_labels.pkl".format(args.dataset), 'rb'))
				
				y_poisoned = pkl_data['split_{}'.format(split)]['{}_percent_poison'.format(poison_ratio)].argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()

				if args.defence:
					# A^2 Y_poisoned
					ad = normalize_adj(sp.csr_matrix(dataset.adj.todense()[dataset.train_ids, :][:, dataset.train_ids])).todense()
					m = torch.tensor(ad).to(device)
					yp_tr = torch.eye(y_poisoned.max()+1).cuda()[y_poisoned[data.train_mask]].to(device)
					y = ((m@m)@yp_tr).cpu() 
				else: 
					y = y_poisoned[data.train_mask].cpu()


			elif args.attack == 'random':

				y_poisoned = poison_labels_random(labels=dataset.dataset_data['labels'], \
												  train_idx=dataset.train_ids, \
												  BUDGET=BUDGET).argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()


			elif args.attack == 'random_bin':

				y_poisoned = poison_labels_random_binary(labels=dataset.dataset_data['labels'], \
												  		 train_idx=dataset.train_ids, \
												  		 BUDGET=BUDGET).argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()

			
			elif args.attack == 'degree':

				y_poisoned = poison_labels_degree(labels=dataset.dataset_data['labels'], \
												  train_idx=dataset.train_ids, \
												  adj=dataset.adj.todense(), \
												  BUDGET=BUDGET).argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()


			elif args.attack == 'degree_bin':

				y_poisoned = poison_labels_degree_binary(labels=dataset.dataset_data['labels'], \
												  		 train_idx=dataset.train_ids, \
												  		 adj=dataset.adj.todense(), \
												  		 BUDGET=BUDGET).argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()

			
			elif args.attack == 'lp':
				y_poisoned = torch.tensor(lp_attack(dataset, BUDGET), dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()


			if args.attack == 'margin_lp' or args.attack == 'meta':
				y_poisoned = data.y.clone()
				y_poisoned[data.train_mask] = y.cuda()


			# Stratified cross validation
			if args.dataset == 'chameleon' or args.dataset == 'squirrel':
				skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0) #10
			else:
				skf = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)


			CV_train_val_set = []
			CV_fold_idx = 0

			if args.setting == 'cv' and (torch.bincount(y) > 2).all():
				for train_index, val_index in skf.split(np.zeros(y.shape[0]), y.cpu()):
					CV_train_val_set.append((train_index, val_index))
			elif args.setting in ['small_val', 'large_val']:
				CV_train_val_set.append([None, None])
			else:
				continue

			best_v_acc = 0
			best_t_acc = None
			best_poisoned_labels = None
			for idx in range(args.samples if args.sampling else 1):
				if args.attack == 'meta':
					y_train_poisoned = y

				sampler = TPESampler(seed=0) #RandomSampler(seed=0)
				model_study = optuna.create_study(direction='maximize', sampler=sampler) 
				if args.hyp_param_tuning:
					model_study.optimize(objective, n_trials=20, n_jobs=1) 
				else:
					model_study.optimize(objective, n_trials=1, n_jobs=1)

				trial = get_best_trial(model_study)
				best_test_acc = trial.user_attrs['Test Accuracy']
				best_val_acc = trial.user_attrs['Val Accuracy']
				test_accs_log = trial.user_attrs['Test Acc Log']
				val_accs_log = trial.user_attrs['Val Acc Log']

				#logging weights
				if args.model == 'GCN':
					conv1_w = trial.user_attrs['conv1']
					conv2_w = trial.user_attrs['conv2']

					weights_log_splits[split] = [conv1_w, conv2_w]


				print("[CV-Fold-{:d}] [sample-{:d}] Split: {:2d} \t Val Acc: {:.2f} Test Acc: {:.2f}".format(CV_fold_idx, idx, split, best_val_acc, best_test_acc))
				logs[f'poison_ratio_{poison_ratio}'][f'split_{split}'][f'CV'] = EasyDict(test_acc= np.array(test_accs_log).tolist(), val_acc= np.array(test_accs_log).tolist())

				if best_val_acc > best_v_acc:
					best_v_acc = best_val_acc
					best_t_acc = best_test_acc

				torch.cuda.empty_cache()
			model.apply(weight_reset)


			test_accs.append(best_t_acc)
			val_accs.append(best_v_acc)
			CV_fold_idx += 1
			
			del model

		if args.wandb:
			wandb.log({ 'poison_ratio': poison_ratio,
						'test_mean': np.mean(test_accs),
						'test_std': np.std(test_accs),
						'val_mean': np.mean(val_accs),
						'val_std': np.std(val_accs)
						})

		logs[f'poison_ratio_{poison_ratio}'][f'avg_res'] = EasyDict(test_mean = np.mean(test_accs), test_std = np.std(test_accs))
	
		print("Test Acc for Budget-{}%: {:.2f} ({:.2f})".format(poison_ratio, np.mean(test_accs), np.std(test_accs)))
	
		weights_log[poison_ratio] = weights_log_splits

	os.makedirs(os.path.join(args.output_dir, run_name), exist_ok=True)
	torch.save(logs,os.path.join(args.output_dir, run_name,'logs.pickle'))
