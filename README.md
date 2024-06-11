# RethinkingLabelPoisoningForGNNs
Official code for RETHINKING LABEL POISONING FOR GNNS: PITFALLS AND ATTACKS [ICLR 2024](https://openreview.net/pdf?id=J7ioefqDPw)

To run our repository, first install the requirements listed in requirements.txt


Run the below commands with configurable parameters as needed:

```
cd meta-label-poisoning

python meta_label.py --dataset {cora_ml, citeseer, pubmed, corafull_pca} --setting cv --random_train_val --hyp_param_tuning --attack {meta, sgcbin, ntk} --model {MLP, GCN, GAT, APPNP, GCN_JKNet, CPGCN}
```

* our code also supports running baselines attacks. This can be done by choosing the attack flag with these options: {MG, lp, lafak, degree, random, random_bin, degree_bin}

* For simulating the threat model where the ground truth labels are not available for test nodes, turn on the yhat flag. Example command with this setting:
```
python meta_label.py --dataset cora_ml --setting cv --random_train_val --hyp_param_tuning --attack sgcbin --model GCN --yhat
```

For meta-binary attacks, turn on the --binary_attack flag and chose the --binary_setting {1,2,3}
For example:
```
python meta_label.py --dataset cora_ml --setting cv --random_train_val --hyp_param_tuning --attack meta --model GCN --yhat --binary_attack --binary_setting 1
```

Additionally, we also provide two notebooks to generate poisoned labels for our LSA family of attacks inside the meta-label-poisoning folder. In specific, the notebook 'SGCpoisonLabels_yhat.ipynb' can be used to create poisoned labels for different SGC attack variants, and the notebook 'SGC_attack_ntk' can be used to create poisoned labels with NTK variants.

For Linear Surrogate Models optimal solution, refer to 'LSA_optimal.ipynb' notebook in meta-label-poisoning folder.

## Citation

```
@inproceedings{
lingam2024rethinking,
title={Rethinking Label Poisoning for {GNN}s: Pitfalls and Attacks},
author={Vijay Lingam and Mohammad Sadegh Akhondzadeh and Aleksandar Bojchevski},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=J7ioefqDPw}
}
```


