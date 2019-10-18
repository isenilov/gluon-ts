# Code for "High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes"

This is a fork of [GluonTS](https://github.com/awslabs/gluon-ts/tree/master) accompanying the paper 
"High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes" accepted to Neurips 2019 
as a poster.

The code has been rewritten in [GluonTS](https://github.com/awslabs/gluon-ts/tree/master), 
we recommend installing it the following way 

```
git clone https://TODO.git
cd gluon-tsts
pip install -e .
```

To run the benchmark,

```
python src/gluonts/multivariate/run_benchmark.py
```

It will take a long time as it loops over every method and dataset however you can easily change the loop to evaluate 
any given given method/dataset (consider using Sagemaker to launch all evaluations in parallel).

The results obtained with this implementation are as follow for CRPS:

estimator | exchange | solar | elec | taxi | wiki
---- | ---- | ---- | ---- | ---- | ----
@LSTMInd | 0.012+/-0.001 | 0.891+/-0.002 | 0.967+/-0.001 | 0.168+/-0.006 | 0.629+/-0.005 | 0.975+/-0.002
@LSTMIndScaling | 0.011+/-0.001 | 0.465+/-0.006 | 0.152+/-0.012 | 0.140+/-0.002 | 0.541+/-0.026 | 0.355+/-0.004@LSTMFR |  |  |  |  | 
@LSTMFRScaling |  |  |  |  | 
@LSTMCOP | 0.010+/-0.000 | 0.670+/-0.029 | 0.249+/-0.008 | 0.379+/-0.001 | 0.474+/-0.000 | 0.305+/-0.005
@GP | 0.063+/-0.001 | 0.941+/-0.006 | 0.980+/-0.002 | 0.263+/-0.002 | 0.666+/-0.035 | 0.962+/-0.008
@GPScaling | 0.042+/-0.001 | 0.456+/-0.048 | 0.091+/-0.006 | 0.141+/-0.001 | 0.373+/-0.019 | 0.327+/-0.033
@GPCOP | 0.009+/-0.000 | 0.429+/-0.011 | 0.090+/-0.004 | 0.113+/-0.001 | 0.344+/-0.003 | 0.244+/-0.004

The model will also be released in GluonTS, this fork is created to keep a version with results as close as possible as 
the one published in the paper. While being close to the submission, the results are not exactly the same as the code 
was rewritten in GluonTS however this version should be very close if not better.

The code to evaluate baselines VAR and GARCH will be released soon.

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