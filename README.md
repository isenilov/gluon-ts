# Code for "High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes"

This is a fork of [GluonTS](https://github.com/awslabs/gluon-ts/tree/master) accompanying the paper 
"High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes" accepted to NeurIPS 2019 
as a poster.

The code has been rewritten in [GluonTS](https://github.com/awslabs/gluon-ts/tree/master), 
we recommend installing it the following way 

```
git clone https://TODO.git
cd gluon-tsts
pip install -e .
```

To run the model:

```
python src/gluonts/multivariate/train_and_plot_predictions.py
```
This will run the model on the selected dataset with the hyperparameters in the paper (see supplementary material for details 
on the GluonTS implementation). 

The results obtained with this implementation are as follow for CRPS:

estimator | exchange | solar | elec | traffic | taxi | wiki
----------|----------|-------|------|---------|------|-----
@GPCOP    | 0.009+/-0.000  |  0.416+/-0.007 |  0.054+/-0.000 |  0.106+/-0.002 |  0.339+/-0.003 | 0.244+/-0.003

The model will also be released in GluonTS, this fork is created to keep a version with results as close as possible as 
the one published in the paper. While being close to the submission, the results are not exactly the same as the code 
was rewritten in GluonTS. However this version should be very close if not better.

## Citing

If the datasets, benchmark, or methods are useful for your research, you can reference the following paper:

```
@article{lowrank_gp_multivariate_neurips,
  title={{High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes}},
  author={Salinas, D. and Bolhke-Schneider M. and Callot L. and Medicco R. and Gasthaus J.},
  journal={International Conference on Neural Information Processing Systems},
  series = {NEURIPS'19},
  year={2019}
}
```
