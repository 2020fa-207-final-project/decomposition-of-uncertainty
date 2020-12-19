## Welcome to decomposition of uncertainty.

### Authors
- Owen Callen
- Gabriel Pestre
- Hayden Sansum
- Nikhil Vanderklaauw

### Purpose 

This project is a review and extension of the work, ["Decomposition of Uncertainty in Bayesian Deep Learning
for Efficient and Risk-sensitive Learning"](http://proceedings.mlr.press/v80/depeweg18a/depeweg18a.pdf) by Stefan Depeweg, Jose Miguel Hernandez-Lobato,
Finale Doshi-Velez, and Steffen Udluft.

### Important components of the repo include: 

#### Final Report.
Our final report can be found [here](https://github.com/2020fa-207-final-project/decomposition-of-uncertainty/blob/master/decomposition_of_uncertainty_report.ipynb).

#### Examples
Our example can be found in the notebooks folder, and they are split into three primary examples.
- Bimodal noise.
- Heteroscedastic noise.
- Wet chicken example (Reinforcement learning context)

For each of the examples we attempted three different sampling techniques.
- Hamiltonian Monte Carlo. 
- No U-turn sampler. 
- Black Box variational inference.


#### Implementations of Bayesian models.
This code can be found in the utils folder.
For neural networks, we describe the dimensions
with the following conventions, unless otherwise noted:
- The input X is generally a dataset with N rows and M features.
  In some cases, we may have a "stack" of R datasets
  (i.e. X may be a R-by-N-by-M tensor instead of the usual N-by-M matrix)
  but this case is not supported by our initial implementations.
- The output Y is generally an N-by-K matrix (for a K-output model)
  but may also be R-by-N-by-K, analogously with the input X.
- The weights W of a single neural network are stored
  as a 1-by-D matrix (for a network with D weights).
  When representing the weights of a set of S models
  (e.g. mutiple samples from a posterior), the weights
  may be stored in an S-by-D matrix. (Additionally,
  our implementation has an internal representation
  that uses a list of weights for each layer, but
  that version need not be exposed to the user.)
- In a latent variable model, the X and Y inputs follow the same
  conventions as above, but L latent features of Gaussian noise
  are appeneded to the input X to form an augmented input X'.
  Typically there is a single latent variable (i.e. L=1),
  but we have made our implementation flexible so that we can
  exeperiment with multiple latent variables.
  Additionally, Gaussian noise is added to each of the output Y
  (potentially with a different variance for each of the K outputs),
  to form a pertured output Y'.
In summary:
- R: number of "stacked" datasets for X and Y (and X' and Y').
- N: number of data points in X and Y (and X' and Y').
- M: number of features in X.
- K: number of outputs in Y.
- S: number of different models in W.
- D: number of weights in each model.
- L: number of latent noise inputs.

## General setup:

The section for it is for anyone interested in running our code.

After Fork in the repo to your local machine, follow these steps to install the Pipenv We used in developing this library.

To setup the env:  
`pip install pipenv`  
`pipenv install --dev`

To install new packages:  
`pipenv install ... # some package`

To run tests:  
`pipenv run pytest`  
or  
`pipenv shell`  
`pytest`  

To enter the shell:  
`pipenv shell`  

To leave the shell:
`exit`  

## Setup on DeepNote:

_Adapted from [this tutorial](https://github.com/sportsdatasolutions/python_project_template#dependencies):_

1. Follow [DeepNote's instructions](https://docs.deepnote.com/integrations/github) for linking a GitHub repo to a DeepNote project.
    - Note: If you are a member of multiple organizations, DeepNote lets you select which one(s) to grant it access to. However, that prompt is only displayed the first time you link DeepNote to GitHub. If you have already authorized GitHub to you DeepNote account and want to grant additional access to the `2020fa-207-final-project` organization, you may need to revoke DeepNote's access and then re-authorize it. To revoke access, go to the [Applications section](https://github.com/settings/applications) of your GitHub account settings, click the `Authorized OAuth Apps` tab, find DeepNote, and revoke its access. Then, go back to DeepNote and try linking the `decomposition-of-uncertainty` repo -- this time you should get a prompt asking you to authorize DeepNote and choose which organizations to grand access to. (You will also need to re-link the repos in your existing projects, as their public keys will have been invalidated when you revoked DeepNote's access.)

2. [Open a DeepNote terminal](https://docs.deepnote.com/features/terminal), which by default will place you in the `~/work` directory. Typing `ls` should show the following files:
- `decomposition-of-uncertainty`: Clone of the GitHub repo.
- `init.ipynb`: Initialization notebook (hidden) -- DeepNote runs it automatically when it boots up an instance.
- `notebook.ipynb`: Default notebook created by DeepNote (can be deleted).

3. Create a symbolic link to the Pipfile so that it looks like it's in the `~/work` directory:
```bash
ln -s ~/work/decomposition-of-uncertainty/Pipfile ~/work/Pipfile
```

4. From the [Environment panel](https://docs.deepnote.com/environment/selecting-hardware), replace the contents of the `init.ipynb` with the following:

```bash
%%bash
# If your project has a 'Pipfile' file, we'll install it here apart from blacklisted packages that interfere with Deepnote (see above).
if test -f Pipfile
  then
    sed -i '/jedi/d;/jupyter/d;' Pipfile
    pip install pipenv
    pipenv install --skip-lock
  else
    pip install pipenv
    pipenv install --skip-lock
fi
```

5. Also from the [Environment panel](https://docs.deepnote.com/environment/selecting-hardware), restart the DeepNote instance. From now on, when DeepNote boots up an instance for this project, it will create the environment defined in the Pipfile.
