# AMS
AMS is a tool to automatically generate AutoML search spaces from
users' weak specifications. A weak specification is defined a set
of API classes to include in the AutoML search space. AMS then
extends this set with complementary classes, functionally-related classes,
and relevant hyperparameters and possible values. This configuration can
then be paired with existing search techniques to generate ML pipelines.
AMS relies on API documentation and corpus of code examples to strengthen
the input weak spec.

# Artifact
You can download a VM (if you are not already using it) from
https://ams-fse.s3.us-east-2.amazonaws.com/ams.ova .

The DOI for this artifact is
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3870818.svg)](https://doi.org/10.5281/zenodo.3870818)


```bash
$ wget https://ams-fse.s3.us-east-2.amazonaws.com/ams.ova
```
The should result in a `.ova` (format version 1.0) that can be imported into
virtualbox or vmware. This image was exported and tested with  Virtualbox version 6.0.


If you do so, you can skip all steps below relating to building and simply
navigate to the `ams` folder and activate the conda environment

```bash
$ cd ~/ams/
$ conda activate ams-env

```

### Docker Container
We have also included a `Dockerfile` that installs conda and sets up the
`ams` dependencies up for you. You may find that easier (more convenient to
use) than your base machine. If so, you can run

```bash
docker build . -t ams-container --memory=8g
```

to build the container. Then you can start it with

```bash
docker run -it --memory=8g ams-container
```

You may want to increase the memory allotted for the docker `run` command as you see fit
(and you may be able to decrease it for the docker `build` command). You may
also find this post useful https://stackoverflow.com/questions/44533319/how-to-assign-more-memory-to-docker-container

After you have started the docker container by executing the `run` command,
you can jump to running
`scripts/folder_setup.sh` (please see the installation section below for more
details).

# Building
AMS *should* run without issues on Ubuntu and Mac OSX (tested on 10.11.6).
If you have issues running, we suggest using the pre-packaged VM.

### Installation
First, you should install `conda`. If you don't have `conda` already,
please install from

https://docs.conda.io/en/latest/miniconda.html

Once you have done so, you can build the conda environment

```bash
$ conda env create -f environment.yml
```

This creates the conda environment `ams-env`.

All scripts/commands should be executed from the root `ams` directory
and with the `ams-env` environment active (i.e. run `conda activate ams-env`
when using AMS).

If you would like to modify the location where data/resources etc are
saved down, you should edit `scripts/folder_setup.sh` accordingly.
These paths are used/referenced throughout the remainder of the setup.
Note that the data folder (`$DATA`) should point to `data/` directory
in the root of the project. You can change this, but we do not recommend it.

You should then install some additional supporting resources by running

```bash
$ bash scripts/setup.sh
```

This may take some amount of time as it has to download/build multiple
third-party tools.


You can verify that your installation has completed successfully by running

```bash
$ python -m core.generate_search_space --help
```

You should see a help message printed to the console.


### Building AMS
To use AMS, AMS first extracts rules for complementary components,
indexes the API documents, and extracts hyperparameter/value frequency
distributions from the code corpus. To build this, just execute

```bash
$ bash scripts/build_ams.sh
```

This may take some time as AMS traverse the API's module hierarchy
and extracts information from the code corpus. You will see messages such as
`Trying <...>` or `Failed <...>`. These can be safely ignored.

Once you are done with this, you are ready to use AMS and shouldn't need
to tweak anything else.


# Known Latency Issue
Loading the SciSpacy language model takes approximately 30 seconds.

```python
spacy_nlp = spacy.load("en_core_sci_lg")
```
This is a known hurdle for language models in Spacy
(see https://github.com/explosion/spaCy/issues/2679 for a similar example)

This latency is a cost to starting up AMS on each new weak specification.
A simple workaround (that has not yet been implemented), is to run
AMS as a server application, with new weak specifications provided as input
from a client. We eschew this for the current artifact as it may complicate
use of AMS for reviewers.


# Using AMS
If you would like to use AMS to generate search spaces for your weak specs,
you can use the script `scripts/use_ams.sh`.

For example,

```bash
$ (bash scripts/use_ams.sh sklearn.linear_model.LogisticRegression sklearn.preprocessing.MinMaxScaler) > config.json
```

produces a strengthened search space for this weak specification.

```bash
$ cat config.json | jq # note we don't install jq
{
  "sklearn.linear_model.SGDClassifier": {
    "loss": [
      "log",
      "hinge"
    ],
    "penalty": [
      "l2",
      "elasticnet",
      "l1"
    ],
    "alpha": [
      1e-05,
      0.0001
    ]
  },
  "sklearn.linear_model.LogisticRegression": {
    "C": [
      100000,
      7,
      100,
      1
    ],
    "penalty": [
      "l1",
      "l2"
    ],
    "class_weight": [
      "auto",
      "balanced",
      null
    ]
  },
  "sklearn.linear_model.RidgeClassifier": {
    "solver": [
      "sag",
      "auto"
    ],
    "tol": [
      0.01,
      0.001
    ]
  },
  "sklearn.preprocessing.StandardScaler": {
    "copy": [
      true,
      false
    ],
    "with_mean": [
      true,
      false
    ],
    "with_std": [
      true
    ]
  },
  "sklearn.linear_model.Log": {},
  "sklearn.preprocessing.MinMaxScaler": {
    "copy": [
      true
    ]
  },
  "sklearn.preprocessing.RobustScaler": {},
  "sklearn.preprocessing.MaxAbsScaler": {},
  "sklearn.preprocessing.Binarizer": {}
}
```



The script hardcodes various AMS choices, which you can modify as desired.
In particular, the script sets:

```bash
NUM_COMPONENTS=4
NUM_ASSOC_RULES=1
ALPHA_ASSOC_RULES=0.5
NUM_PARAMS=3
NUM_PARAM_VALUES=3
```

`NUM_COMPONENTS` refers to the number of functionally related components
to add (at most) per component in the weak spec. `NUM_ASSOC_RULES` refers
to the number of complementary components to add (at most) per component
in the weak spec. `ALPHA_ASSOC_RULES` combines a rule's normalized-PMI
and support fraction to obtain a single score for an association rule,
we used 0.5 in our evaluations but you can modify if you'd like. Please
see the paper for details. `NUM_PARAMS` is the (max) count of hyperparameters to
include in the search space for each component in the extended specification,
and `NUM_PARAM_VALUES` is the (max) number of possible values (in addition
to the default value) for each hyperparameter in the extended search space.

# Using an AMS generated search space
AMS produces a search space in JSON format. This configuration can be
read in and used directly with TPOT, or with the random search procedure
used for AMS evaluation.

For example, we first generate the search space from the prior example
and dump it into a text file

```bash
$ (bash scripts/use_ams.sh sklearn.linear_model.LogisticRegression sklearn.preprocessing.MinMaxScaler) > config.json
```

We then launch the python interpreter, read in the configuration,
and show how it can be used on a generated dataset

```python
$ python
import json
import tpot
import sklearn.datasets
from core.search import RandomSearch

X, y = sklearn.datasets.make_classification(100, 10)
config = json.load(open("config.json", "r"))

# GP-based search
clf_tpot = tpot.TPOTClassifier(max_time_mins=1, config_dict=config, verbosity=3)

# Random search
clf_rand = RandomSearch(max_time_mins=1, max_depth=3, config_dict=config)


clf_tpot.fit(X, y)
clf_rand.fit(X, y)
```


# Reproducing FSE evaluation
You can reproduce FSE experiments and figures by using scripts in
`scripts/fse/`. As others, these should be run from the root AMS directory.
Given that some of the experiments explained below take on the order of days
to run on a well-provisioned machine, we provide a download of our experimental
results. (If you are using the artifact VM, these results have already been
loaded and you can skip the following step).

### Datasets
Note that we have included all necessary datasets directly in the repository
(and artifact), as the library that packages these datasets has implemented
breaking changes with no backwards compatibility.

### Downloading Results
If you want to download the results, you can run

```bash
$ bash scripts/fse/download_results.sh
```

This will download results from an AWS S3 bucket and will place results in
`$RESULTS`, `$ANALYSIS_DIR`, and `$DATA`. In particular,

`${RESULTS}` will now contain folders of the form `q[0-9]+`, one for each of
the 15 weak specifications in our experiments. In it, you will find the
weak specification (`simple_config.json`),
the specification with expert-defined hyperparameters (`simple_config_with_params_dict.json`), and the AMS-generated search space
(`gen_config.json`).

The folder `random` contains results (again organized by weak specification
experiment) when using random search, while `tpot` contains results
when using genetic programming (the TPOT tool).

`rule-mining/` has experimental results for the complementary component
experiments.

`${DATA}/corpus-size*` contains objects derived by AMS from a subsample
of the corpus.

#### Tables and Figures

The folder `$ANALYSIS_DIR` (`analysis_output` in the VM) holds figures/tables produced in analysis
and used in the paper. In particular,

* Table 2: `rules/roles.tex`
* Figure 3: `rules/precision.pdf`
* Figure 4: `relevance/plot.pdf`
* Figure 5:
    - (a) `hyperparams/num_params_tuned.pdf`
    - (b) `hyperparams/distance_params_tuned.pdf`
    - (c) `hyperparams/num_param_values.pdf`
* Figure 6: `hyperparams/perf.pdf`
* Figure 7: `performance/combined_wins.pdf`
* Figure 8: `tpot-sys-ops/combined.pdf` (includes extended examples)
* Figure 9:
    - (a) `corpus-size/hyperparameters.pdf`
    - (b) `corpus-size/hyperparameter_values.pdf`
    - (c) `corpus-size/num_mined_rules.pdf`
    - (d) `corpus-size/jaccard_mined_rules.pdf`

(Tip: To open PDFs from the terminal in the Ubuntu VM, you can use `xdg-open file.pdf`)

### Scripts
`bash scripts/fse/reproduce_complementary_experiments.sh` reproduces experimental results relating
to extraction of complementary components to add to weak specifications. These
experiments should take on the order of a couple of hours to run.

`bash scripts/fse/reproduce_functional_related_experiments.sh` generates the data used for
manual annotation of functionally related components. We have already included
our manually annotated results as part of the artifact, so running this script
will prompt you to confirm before overwriting those with (unannotated) data.

`bash scripts/fse/reproduce_performance_experiments.sh` generates search space configurations
and evaluates them against our comparison baselines (weak spec, weak spec + search,
and expert + search). Running these experiments from scratch *takes on the order
of 1-2 days on a machine with 30 cores*. Given this computational burden, we have
also included our results in the artifact.

`bash scripts/fse/reproduce_corpus_size.sh` generates subsampled versions of
our code corpus, and rebuilds portions of AMS that rely on code examples
(i.e. hyperparameter mining and complementary component mining). Running
these experiments from scratch *takes on the order of 2 hours on a machine
with 30 cores*. Given this computational burden, we have also included our
results in the artifact. Note that in contrast to other scripts, this
is "creating new data" and as such the outputs are stored in `${DATA}`,
following the naming convention `corpus-size-${corpus_size}-iter-${corpus_iter}/`
where `${corpus-size}` is the downsampling ratio (e.g. 0.1) and
`${corpus-iter}` is the iteration index (e.g. 1)
as we repeat the downsampling 5 times per ratio.

`bash scripts/fse/reproduce_analysis.sh` generates figures from the outputs of the prior
3 scripts and also runs some additional (~ 1 hour execution) experiments to
characterize the hyperparameters found in our code corpus.

The figures/tables are generated and saved to `$ANALYSIS_DIR`
(set to `analysis_output/`, if not modified in `folder_setup.sh`).
Please see the prior section for details on figure/table mappings.



# Codebase Overview
We provide a short overview of the AMS codebase:

* `core/` contains the main tool logic:
  - `extract_sklearn_api.py`: traverse sklearn modules to find classes
  to import and represent with embeddings (also has stuff on default parameters)
  - `nlp.py`: helper functions to parse/embed natural language
  - `code_to_api.py`: takes a code specification and maps it to possibly
  related API components using pre-trained embeddings
  - `extract_kaggle_scripts.py`: filter down meta-kaggle to find useful scripts
  (i.e. those that import target scikit-learn library)
  - `extract_parameters.py`: (light) parse of python scripts from kaggle
  to extract calls to APIs and their parameters
  - `summarize_parameters.py`: tally up frequent parameter names/values
  by API component
  - `generate_search_space.py`: given weak spec (code/nl) generate
  search space dictionary
  - `search.py`: various search strategies and helpers


* `experiments/` contains all code to run experiments:
  - `download_datasets.py`: download benchmark datasets and cache (in case working without internet later on)
  - `generate_experiment.py`: generate experiment configurations based on some predefined components of interest
  - `simple_pipeline.py`: compile weak spec directly into a sklearn pipeline for benchmarking
  - `run_experiment.py`: driver to run different search strategies/configurations
  - `build_corpus_size_experiment.py`: driver to run corpus size experiments, downsamples corpus and rebuilds portions of AMS that use that data


* `analysis/`: contains code to run analysis on experiment outputs and conduct
additional characterization of our data.
  - `annotation_rules_component_relevance.md`: details our annotation guidelines for manually assessing functionally related components
  - `association_rules_analysis.py`: evaluates the rules used to extend specifications with complementary components
  - `combined_wins_plot.py`: is a utility to combine win counts into a single plot
  - `distribution_hyperparameters.py`: characterizes hyperparameters found in our code corpus
  - `frequency_operators.py`: compute distribution of components in pipelines
  - `performance_analysis.py`: compute table/plots of wins from performance experiments data
  - `pipeline_to_tree.py`: convert pipeline from API into a tree (easier to analyze) utility
  - `relevance_markings.py`: create plot of functionally related components' manual annotation results
  - `corpus_size_analysis.py`: create plots for impact of corpus size
  - `utils.py`: misc utils


# Contact
Feel free to email Jose Cambronero (jcamsan@mit.edu) with questions.
