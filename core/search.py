from datetime import datetime

import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.pipeline import Pipeline
import sklearn.base
import tqdm
import traceback
import warnings
warnings.filterwarnings(action='ignore')

from core import mp_utils
from core.utils import get_component_constructor


def get_random_state(st):
    if st is None:
        st = np.random.RandomState(st)
    return st


def randint(low, high, random_state=None):
    return get_random_state(random_state).randint(low=low, high=high)


def randunif(random_state=None):
    return get_random_state(random_state).uniform()


def randchoice(options, random_state=None):
    return get_random_state(random_state).choice(options)


def choose_rand_depth(max_depth, random_state=None):
    return randint(1.0, max_depth + 1, random_state=random_state)


def choose_rand_component(components, random_state=None):
    return randchoice(components, random_state)


def remove_list_values(d):
    # need to be able to frozenset/hash stuff
    clean_d = {}
    for k, v in d.items():
        if isinstance(v, (list, set)):
            v = tuple(v)
        clean_d[k] = v
    return clean_d


def choose_rand_hyperparam_config(config, random_state=None):
    chosen_config = {}
    for config_key, config_values in config.items():
        if isinstance(config_values, dict):
            # we don't handle nested hyperparameters
            continue
        chosen_config[config_key] = randchoice(config_values, random_state)
    # return immutable version so we can hash
    chosen_config = remove_list_values(chosen_config)
    return frozenset(chosen_config.items())


def hierarchical_random_generate_pipeline(
    acc,
    depth,
    classifier_config_dict,
    transformer_config_dict,
    random_state=None,
):
    if depth == 1:
        config = classifier_config_dict
    else:
        config = transformer_config_dict

    comp = choose_rand_component(
        list(config.keys()),
        random_state=random_state,
    )
    hyper = choose_rand_hyperparam_config(
        config[comp],
        random_state=random_state,
    )
    chosen = (comp, hyper)
    acc.append(chosen)

    if depth == 1:
        # make it immutable so we can hash it
        acc = tuple(acc)
        return acc
    else:
        return hierarchical_random_generate_pipeline(
            acc,
            depth - 1,
            classifier_config_dict,
            transformer_config_dict,
            random_state=random_state
        )


def choose_rand_pipeline(
    max_depth,
    classifier_config_dict,
    transformer_config_dict,
    random_state=None,
):
    if len(transformer_config_dict) == 0:
        # nothing to do but apply classifier
        depth = 1
    else:
        depth = choose_rand_depth(max_depth, random_state=random_state)
    acc = []
    return hierarchical_random_generate_pipeline(
        acc,
        depth,
        classifier_config_dict,
        transformer_config_dict,
        random_state=random_state,
    )


def compile_pipeline(comps_with_param_tuples):
    steps = []
    for ix, (comp, param_tuples) in enumerate(comps_with_param_tuples):
        constructor = get_component_constructor(comp)
        params = dict(param_tuples)
        obj = constructor(**params)
        steps.append(("step_{}".format(ix), obj))
    return Pipeline(steps)


def separate_classifier_transformer_configs(search_config):
    classifier_config = {}
    transformer_config = {}
    for comp, comp_config in search_config.items():
        obj = get_component_constructor(comp)
        if sklearn.base.is_classifier(obj):
            classifier_config[comp] = comp_config
        if hasattr(obj, "transform") or hasattr(obj, "fit_transform"):
            transformer_config[comp] = comp_config
    return classifier_config, transformer_config


class PipelineException(Exception):
    # used to track hard to debug exceptions
    # (want to know what pipeline raised
    # so can know if reasonable to catch/ignore root exception)
    # hard to keep track of otherwise
    def __init__(self, message, detailed_message, pipeline=None):
        super().__init__(message)
        self.detailed_message = detailed_message
        self.pipeline = pipeline


class CustomSearch(object):
    def __init__(
        self,
        config_dict,
        max_depth,
        max_time_mins=1,
        max_time_mins_per_pipeline=1,
        max_retries=100,
        cv=5,
        scoring=None,
        random_state=None,
        **params,
    ):
        self.config_dict = config_dict
        if isinstance(self.config_dict, dict):
            self.classifier_config_dict, self.transformer_config_dict = separate_classifier_transformer_configs(
                config_dict
            )
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.fitted_pipeline_ = None
        self.cv = 5
        self.scoring = scoring
        self.max_time_mins = max_time_mins
        self.max_time_mins_per_pipeline = max_time_mins_per_pipeline
        self.random_state = np.random.RandomState(random_state)
        self.best_score = None
        self._start_datetime = None
        self.pipelines_tried_ = set([])
        self.time_step = 1.0

    def time_so_far(self):
        return (datetime.now() - self._start_datetime).total_seconds() / 60.

    def fit(self, X, y):
        self._start_datetime = datetime.now()
        self.search_loop(X, y)
        # fit best on full dataset
        self.fitted_pipeline_.fit(X, y)
        print("Executed for", self.time_so_far(), "minutes")
        return self

    def search_loop(self, X, y):
        self.pipelines_tried_ = set()
        ct = 0
        pbar_limit = 100
        pbar = tqdm.tqdm(total=pbar_limit)
        while True:
            try:
                total_mins_elapsed = self.time_so_far()
                if total_mins_elapsed >= self.max_time_mins:
                    raise KeyboardInterrupt
                self._generate_and_validate_pipeline(
                    X,
                    y,
                    timeout=max(int(self.max_time_mins_per_pipeline * 60), 1),
                )
                ct += 1
                pbar.update(1)
                if ct % pbar_limit == 0:
                    pbar = tqdm.tqdm(total=pbar_limit)
                    print("Evaluated {} pipelines so far".format(ct))
            except KeyboardInterrupt:
                print(
                    "Timed out: {:.2f}/{:.2f} min.".format(
                        self.time_so_far(),
                        self.max_time_mins,
                    )
                )
                self.pipelines_tried_ = list(self.pipelines_tried_)
                if self.fitted_pipeline_ is None:
                    pbar.close()
                    raise TimeoutError("No pipelines fitted in time")
                else:
                    pbar.close()
                    return self
        pbar.close()
        return self

    def _fetch_new_pipeline(self):
        raise NotImplementedError("Implement in subclass")

    def _generate_and_validate_pipeline(self, X, y, timeout=None):
        # default case: pipeline fails
        pipeline = None
        pipeline_def = None
        mean_score = np.nan

        try:
            # pipeline should be fetched within try/catch
            # in case a constructor fails (e.g. requires positional argument)
            pipeline, pipeline_def = self._fetch_new_pipeline()
            if pipeline is None:
                print(
                    "Sampled {} repeated pipelines in a row".format(
                        self.max_retries
                    )
                )
                raise KeyboardInterrupt("Terminating early due to retries")
            # a remote process, with timeout
            # the timeout thing with stopit (thread-based) is not working out...
            results = mp_utils.run(
                timeout,
                sklearn.model_selection.cross_validate,
                pipeline,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=True,
            )
            mean_score = np.mean(results["test_score"])
            self.update_pipeline(pipeline, pipeline_def, mean_score)
        except KeyboardInterrupt as err:
            # toss this back up to next level, so we can terminate search
            # loop
            raise err
        except (
                TimeoutError,
                ValueError,
                TypeError,
                ZeroDivisionError,
                IndexError,
                AttributeError,
                MemoryError,
                ImportError,
                mp_utils.TimeoutError,
                mp_utils.mp.pool.MaybeEncodingError,
        ) as err:
            detailed_msg = traceback.format_exc()
            print("Swallowing error: " + detailed_msg)
            mean_score = np.nan
            self.update_pipeline(pipeline, pipeline_def, mean_score)
        except PipelineException as err:
            # just re raise, don't wrap
            raise err
        except Exception as err:
            # wrap everything else, so we can have a chance at debugging
            detailed_msg = traceback.format_exc()
            raise PipelineException(err, detailed_msg, pipeline=pipeline)

    def update_pipeline(self, pipeline, pipeline_def, mean_score):
        if np.isnan(mean_score):
            # don't set a new best pipeline if failed
            return
        if self.best_score is None or mean_score > self.best_score:
            self.fitted_pipeline_ = pipeline
            self.best_score = mean_score

    def predict(self, X):
        return self.fitted_pipeline_.predict(X)

    def predict_proba(self, X):
        return self.fitted_pipeline_.predict_proba(X)


class RandomSearch(CustomSearch):
    def __init__(self, **params):
        super().__init__(**params)

    def _fetch_new_pipeline(self):
        pipeline_def = None
        found_new_pipeline = False

        for _ in range(self.max_retries):
            pipeline_def = choose_rand_pipeline(
                self.max_depth,
                self.classifier_config_dict,
                self.transformer_config_dict,
                random_state=self.random_state,
            )
            # stops iterating if we found an otherwise untested pipeline
            if pipeline_def not in self.pipelines_tried_:
                found_new_pipeline = True
                break

        if found_new_pipeline:
            self.pipelines_tried_.add(pipeline_def)
            pipeline = compile_pipeline(pipeline_def)
            return pipeline, pipeline_def
        else:
            return None, None


def generate_predefined_pipeline_random_hyperpams(
    config_list, random_state=None
):
    pipeline_def = []
    for step in config_list:
        component = list(step.keys())[0]
        params = step[component]
        chosen_params = choose_rand_hyperparam_config(params)
        pipeline_def.append((component, chosen_params))
    return tuple(pipeline_def)


class DefinedPipelineRandomHyperParamSearch(CustomSearch):
    # Only randomly searches for hyper-parameters
    # but the structure and components of pipeline are
    # pre-defined in the configuration
    def __init__(self, **params):
        super().__init__(**params)

    def _fetch_new_pipeline(self):
        found_new_pipeline = False
        for _ in range(self.max_retries):
            pipeline_def = generate_predefined_pipeline_random_hyperpams(
                self.config_dict,
                random_state=self.random_state,
            )
            # stops iterating if we found an otherwise untested pipeline
            if pipeline_def not in self.pipelines_tried_:
                found_new_pipeline = True
                break

        if found_new_pipeline:
            self.pipelines_tried_.add(pipeline_def)
            pipeline = compile_pipeline(pipeline_def)
            return pipeline, pipeline_def
        else:
            return None, None


# Occasionally TPOT fails
# so rather than waste everything, we just mark
# that iteration of CV as a failure and move on
class FailedOptim(object):
    def __init__(
        self,
        error,
        X=None,
        y=None,
        search=None,
        default_prob=0.1,
        default_label=0,
    ):
        self.error = error
        self.fitted_pipeline_ = None
        # save a copy of the data that raised error
        # for debugging
        self.X = X
        self.y = y
        self.search = search

        self.default_prob = default_prob
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y[y.columns[0]]
            # set first value as default label, in case types are different
            # than integer
            default_label = y[0]
        self.default_label = default_label

    def predict_proba(self, X):
        return np.repeat(self.default_prob, X.shape[0])

    def predict(self, X):
        return np.repeat(self.default_label, X.shape[0])


def add_noise(y, noise):
    if noise <= 0:
        return y
    y = y.copy()
    unique_labels = np.unique(y)
    n = len(y)
    size = int(noise * n)
    ixs = np.random.choice(np.arange(0, n), size=size, replace=False)
    for ix in ixs:
        y_true = y[ix]
        options = [l for l in unique_labels if l != y_true]
        y_noise = np.random.choice(options, size=1)
        y[ix] = y_noise
    return y


class RobustSearch(sklearn.base.BaseEstimator):
    def __init__(self, search_model, noise=None):
        mp_utils.init_mp()
        self.search_model = search_model

    def fit(self, X, y):
        try:
            self.search_model.fit(X, y)
        except (Exception, RuntimeError, TimeoutError) as err:
            error_msg = traceback.format_exc()
            print(error_msg)
            self.failed_model = self.search_model
            self.search_model = FailedOptim(
                err,
                X=X,
                y=y,
                search=self.search_model,
            )

    def __getattr__(self, attr_name):
        return getattr(self.search_model, attr_name)
