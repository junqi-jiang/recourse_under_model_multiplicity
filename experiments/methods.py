import copy
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from scipy.stats import binom_test
from sklearn.neighbors import KDTree
from tabulate import tabulate


class Datasets:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        if len(X_test) >= 1000:
            d_idxs = np.random.randint(0, len(X_test) - 1, 1000)
            self.X_test = pd.DataFrame(data=X_test.values[d_idxs], columns=X_test.columns)
            self.y_test = pd.DataFrame(data=y_test.values[d_idxs], columns=y_test.columns)


def get_cfx_for_x(model, x, original_class, X_train1):
    X_pool = X_train1.values[model.predict(X_train1) != original_class]
    tree = KDTree(X_pool, leaf_size=40)
    idxs = np.array(tree.query(x.reshape(1, -1), k=1)[1]).flatten()
    tree_ces = X_pool[idxs].flatten()
    return tree_ces


class ModelsAndScores:
    def __init__(self, all_models, accuracies, simplicity):
        self.all_models = all_models
        self.accuracies = accuracies
        self.simplicity = simplicity
        self.model_len = len(all_models)


class EnsembleMethod:
    def __init__(self, x, models_and_scores, method="MV"):
        self.ensemble_idxs = []
        self.pred_res = -1
        self.x = x
        self.res_all = None
        self.multiple_ans = False
        if method == "MV":
            self.mv(models_and_scores)
        if method == "SE":
            self.se(models_and_scores)

    def mv(self, models_and_scores):
        res = np.zeros((models_and_scores.model_len))
        for i, m in enumerate(models_and_scores.all_models):
            res[i] = m.predict(self.x.reshape(1, -1))[0]
        self.res_all = res
        predictions, counts = np.unique(res, return_counts=True)
        first_class, first_count = predictions[np.argpartition(counts, -1)[-1]], counts[np.argpartition(counts, -1)[-1]]
        if len(predictions) != 1:
            _, second_count = predictions[np.argpartition(counts, -2)[-2]], counts[np.argpartition(counts, -2)[-2]]
            if first_count == second_count:
                self.multiple_ans = True
        self.pred_res = first_class
        self.ensemble_idxs = np.where(res == self.pred_res)[0]

    def se(self, models_and_scores, alpha=0.05):
        res = np.zeros((len(models_and_scores.all_models)))
        for i, m in enumerate(models_and_scores.all_models):
            res[i] = m.predict(self.x.reshape(1, -1))[0]
        predictions, counts = np.unique(res, return_counts=True)
        self.res_all = res
        first_class, first_count = predictions[np.argpartition(counts, -1)[-1]], counts[np.argpartition(counts, -1)[-1]]
        if len(predictions) == 1:  # all classifiers agree
            self.pred_res = first_class
            self.ensemble_idxs = np.where(res == self.pred_res)[0]
            return
        _, second_count = predictions[np.argpartition(counts, -2)[-2]], counts[np.argpartition(counts, -2)[-2]]
        if binom_test([first_count, second_count]) > alpha:  # abstain
            return
        self.pred_res = first_class
        self.ensemble_idxs = np.where(res == self.pred_res)[0]

    def ours(self, x):
        pass


class CounterfactualSelection:
    def __init__(self, d, x, models_and_scores, ensemble_method):

        # attributes
        self.cfxs_all = None
        self.cfxs_pred = None
        self.final_cfx_idxs = []
        if len(ensemble_method.ensemble_idxs) == 0:
            return

        # compute cfxs for all models
        self.cfxs_all = np.zeros((models_and_scores.model_len, len(x)))
        for i, m in enumerate(models_and_scores.all_models):
            self.cfxs_all[i] = get_cfx_for_x(m, x, ensemble_method.res_all[i], d.X_train)

        # evaluate each model's cfx on all other models
        self.cfxs_pred = np.zeros((models_and_scores.model_len, models_and_scores.model_len))
        for i, m in enumerate(models_and_scores.all_models):
            self.cfxs_pred[i] = m.predict(self.cfxs_all)

    def all_cfxs(self, ensemble_method):  # c
        self.final_cfx_idxs = np.arange(len(ensemble_method.res_all))

    def valid_for_all(self, ensemble_method):  # cvv
        for i in range(len(self.cfxs_all)):
            valid = True
            for j, cfx_res in enumerate(self.cfxs_pred[:, i]):
                if cfx_res == ensemble_method.res_all[j]:
                    valid = False
                    break
            if valid:
                self.final_cfx_idxs.append(i)

    def valid_for_agreeing(self, ensemble_method):  # cv
        for i in range(len(self.cfxs_all)):
            valid = 0
            for j, cfx_res in enumerate(self.cfxs_pred[:, i]):
                if j in ensemble_method.ensemble_idxs and cfx_res != ensemble_method.res_all[j]:
                    valid += 1
            if valid == len(ensemble_method.ensemble_idxs):
                self.final_cfx_idxs.append(i)

    def from_agreeing(self, ensemble_method):  # ca
        self.final_cfx_idxs = ensemble_method.ensemble_idxs

    def valid_for_all_from_agreeing(self, ensemble_method):  # cvva
        self.valid_for_all(ensemble_method)
        self.final_cfx_idxs = [i for i in self.final_cfx_idxs if i in ensemble_method.ensemble_idxs]

    def valid_for_agreeing_from_agreeing(self, ensemble_method):  # cva
        self.valid_for_agreeing(ensemble_method)
        self.final_cfx_idxs = [i for i in self.final_cfx_idxs if i in ensemble_method.ensemble_idxs]

    def run(self, method, ensemble_method):
        # run
        self.final_cfx_idxs = []
        if method == "c":
            self.all_cfxs(ensemble_method)
        if method == "cvv":
            self.valid_for_all(ensemble_method)
        if method == "cv":
            self.valid_for_agreeing(ensemble_method)
        if method == "ca":
            self.from_agreeing(ensemble_method)
        if method == "cvva":
            self.valid_for_all_from_agreeing(ensemble_method)
        if method == "cva":
            self.valid_for_agreeing_from_agreeing(ensemble_method)


class EnsembleAndCounterfactualForOneInput:
    def __init__(self, d, x, models_and_scores, ensemble_method_name="MV", eval_ours=False):
        self.ensemble_idxs = []
        self.cfx_idxs_c = []
        self.cfx_idxs_cvv = []
        self.cfx_idxs_cv = []
        self.cfx_idxs_ca = []
        self.cfx_idxs_cvva = []
        self.cfx_idxs_cva = []
        ensemble = EnsembleMethod(x, models_and_scores, method=ensemble_method_name)
        self.ensemble_idxs = ensemble.ensemble_idxs
        self.pred_res = ensemble.pred_res
        self.res_all = ensemble.res_all
        cfx_selection = CounterfactualSelection(d, x, models_and_scores, ensemble)
        self.cfxs_pred = cfx_selection.cfxs_pred
        self.cfxs_all = cfx_selection.cfxs_all
        cfx_selection.run("ca", ensemble)
        self.multiple_ans = ensemble.multiple_ans
        self.cfx_idxs_ca = copy.copy(cfx_selection.final_cfx_idxs)
        if ensemble_method_name == "MV" and not eval_ours:
            #cfx_selection.run("c", ensemble)
            #self.cfx_idxs_c = copy.copy(cfx_selection.final_cfx_idxs)
            #cfx_selection.run("cvv", ensemble)
            #self.cfx_idxs_cvv = copy.copy(cfx_selection.final_cfx_idxs)
            #cfx_selection.run("cv", ensemble)
            #self.cfx_idxs_cv = copy.copy(cfx_selection.final_cfx_idxs)
            #cfx_selection.run("cvva", ensemble)
            #self.cfx_idxs_cvva = copy.copy(cfx_selection.final_cfx_idxs)
            cfx_selection.run("cva", ensemble)
            self.cfx_idxs_cva = copy.copy(cfx_selection.final_cfx_idxs)


class ArgumentativeEnsemble:
    def __init__(self):
        pass


class Experiment:
    # one set of experiments: given certain number of models and the test dataset, get one set of results
    def __init__(self, d, models_and_scores, dataset_name="credit", run_id=0):
        # attributes
        self.d = d
        self.models_and_scores = models_and_scores
        self.dataset_name = dataset_name
        self.run_id = run_id
        # for eval: acc, complexity, model len, cfx len, cfx val
        self.se_scores = []
        self.mvc_scores = []
        self.mvca_scores = []
        self.mvcvv_scores = []
        self.mvcvva_scores = []
        self.mvcv_scores = []
        self.mvcva_scores = []
        self.ours_scores = []
        self.ours_a_scores = []
        self.ours_s_scores = []
        self.ours_as_scores = []
        self.single_model_avg_acc = 0

    def eval(self):
        # get acc, complexity, model len
        mv_acc = 0
        mv_mlen = 0
        mv_msimp = 0
        mv_multiple_ans = 0
        mvca_cfxlen, mvca_cfxval = 0, 0
        # the next four options might end up with no cfx
        mvcva_cfxlen, mvcva_cfxval, mvcva_abstain = 0, 0, 0
        test_len = self.d.X_test.shape[0]
        # open file for registering commands for solving argumentative ensemble
        f_command = open(f"bafs_{self.dataset_name}/commands{self.run_id}.txt", 'w')
        for i in tqdm(range(test_len), position=0, leave=True):
            x = self.d.X_test.values[i]
            # get ensembles and counterfactuals
            mv_one = EnsembleAndCounterfactualForOneInput(self.d, x, self.models_and_scores, ensemble_method_name="MV")

            # get accuracy
            if int(mv_one.pred_res) == int(self.d.y_test.values[i]):
                mv_acc += 1

            # get model simplicity and len
            mv_mlen += len(mv_one.ensemble_idxs)

            # get mv cfx vals
            # c
            mvc_cfxlen0, mvc_cfxval0, mv_msimp0, _ = self.eval_cfxs_for_one(mv_one.ensemble_idxs, mv_one.cfx_idxs_c,
                                                                            mv_one.cfxs_pred,
                                                                            self.models_and_scores.simplicity,
                                                                            mv_one.res_all)
            mv_msimp += mv_msimp0
            # ca
            mvca_cfxlen0, mvca_cfxval0, _, _ = self.eval_cfxs_for_one(mv_one.ensemble_idxs, mv_one.cfx_idxs_ca,
                                                                      mv_one.cfxs_pred,
                                                                      self.models_and_scores.simplicity, mv_one.res_all)
            mvca_cfxlen += mvca_cfxlen0
            mvca_cfxval += mvca_cfxval0
            # cva
            mvcva_cfxlen0, mvcva_cfxval0, _, mvcva_abstain0 = self.eval_cfxs_for_one(mv_one.ensemble_idxs,
                                                                                     mv_one.cfx_idxs_cva,
                                                                                     mv_one.cfxs_pred,
                                                                                     self.models_and_scores.simplicity,
                                                                                     mv_one.res_all)
            mvcva_cfxlen += mvcva_cfxlen0
            mvcva_cfxval += mvcva_cfxval0
            mvcva_abstain += mvcva_abstain0
            if mv_one.multiple_ans:
                mv_multiple_ans += 1
            # execute ours: create baf only
            get_baf(self.models_and_scores, mv_one.res_all, mv_one.cfxs_pred, input_idx=i, run_id=self.run_id,
                    acc_preferred=False, simp_preferred=False, as_preferred=False, dataset_name=self.dataset_name)
            get_baf(self.models_and_scores, mv_one.res_all, mv_one.cfxs_pred, input_idx=i, run_id=self.run_id,
                    acc_preferred=True, simp_preferred=False, as_preferred=False, dataset_name=self.dataset_name)
            get_baf(self.models_and_scores, mv_one.res_all, mv_one.cfxs_pred, input_idx=i, run_id=self.run_id,
                    acc_preferred=False, simp_preferred=True, as_preferred=False, dataset_name=self.dataset_name)
            get_baf(self.models_and_scores, mv_one.res_all, mv_one.cfxs_pred, input_idx=i, run_id=self.run_id,
                    acc_preferred=False, simp_preferred=False, as_preferred=True, dataset_name=self.dataset_name)
            # create commands
            f_command.write(
                f"clingo bafs_{self.dataset_name}/run{self.run_id}/bafmm_{i}.af baf.dl filter.lp 0 | Out-File -FilePath bafs_{self.dataset_name}/run{self.run_id}/output_{i}.txt\n")
            f_command.write(
                f"clingo bafs_{self.dataset_name}/run{self.run_id}/bafmm_{i}_acc.af baf.dl filter.lp 0 | Out-File -FilePath bafs_{self.dataset_name}/run{self.run_id}/output_{i}_acc.txt\n")
            f_command.write(
                f"clingo bafs_{self.dataset_name}/run{self.run_id}/bafmm_{i}_simp.af baf.dl filter.lp 0 | Out-File -FilePath bafs_{self.dataset_name}/run{self.run_id}/output_{i}_simp.txt\n")
            f_command.write(
                f"clingo bafs_{self.dataset_name}/run{self.run_id}/bafmm_{i}_as.af baf.dl filter.lp 0 | Out-File -FilePath bafs_{self.dataset_name}/run{self.run_id}/output_{i}_as.txt\n")
        f_command.close()
        # for eval: acc (abstain as error), complexity, model len, cfx len, cfx val, abstention rate
        self.mvca_scores = [mv_acc / test_len, mv_msimp / test_len,
                            mv_mlen / test_len / self.models_and_scores.model_len,
                            mvca_cfxlen / test_len / self.models_and_scores.model_len,
                            mvca_cfxval / test_len, 0, mv_multiple_ans / test_len]
        self.mvcva_scores = [mv_acc / test_len, mv_msimp / test_len,
                             mv_mlen / test_len / self.models_and_scores.model_len,
                             mvcva_cfxlen / (
                                         test_len - mvcva_abstain) / self.models_and_scores.model_len if test_len - mvcva_abstain != 0 else -1,
                             mvcva_cfxval / (test_len - mvcva_abstain) if test_len - mvcva_abstain != 0 else -1,
                             mvcva_abstain / test_len, mv_multiple_ans / test_len]

        # get single model accuracy
        for m in self.models_and_scores.all_models:
            self.single_model_avg_acc += accuracy_score(m.predict(self.d.X_test), self.d.y_test)
        self.single_model_avg_acc /= len(self.models_and_scores.all_models)

    def eval_cfxs_for_one(self, ensemble_idxs, cfxs_idxs, cfxs_pred, msimp_scores, res):
        # get cfx len, cfx val, model simplicity score. now have made sure ensemble_idxs are not empty, but cfxs idxs
        # might be empty
        if len(ensemble_idxs) == 0:
            return 0, 0, 0, 1
        msimp = 0
        for i in ensemble_idxs:
            msimp += msimp_scores[i]
        msimp /= len(ensemble_idxs)
        if len(cfxs_idxs) == 0:
            return 0, 0, msimp, 1
        cfx_len = len(cfxs_idxs)
        cfx_val_over_all_cfxs = 0
        for i in cfxs_idxs:
            cfx_val_over_all_cfxs += (
                    np.sum(res[ensemble_idxs].flatten() != cfxs_pred[ensemble_idxs, i].flatten()) / len(
                ensemble_idxs))
        cfx_val_over_all_cfxs /= len(cfxs_idxs)
        return cfx_len, cfx_val_over_all_cfxs, msimp, 0

    def eval_ours_one(self, acc_preferred=False, simp_preferred=False, as_preferred=False):
        test_len = self.d.X_test.shape[0]
        model_len = 0
        res = np.zeros((test_len))
        msimp = 0
        same_as_mv_count = 0
        multiple_ans_count = 0
        multiple_same_class_count = 0
        # for i in tqdm(range(test_len), position=0, leave=True):
        for i in range(test_len):
            x = self.d.X_test.values[i]
            ext_m_full, ext_c_full, ext_all_full, _ = get_extensions_from_baf(i, run_id=self.run_id,
                                                                              acc_preferred=acc_preferred,
                                                                              simp_preferred=simp_preferred,
                                                                              as_preferred=as_preferred, dataset_name=self.dataset_name)
            ext_m, ext_c, ext_all, ans_status, same_as_mv = get_m_preferred_ext(ext_m_full, ext_c_full, ext_all_full, x,
                                                                            self.models_and_scores.all_models)
            if same_as_mv:
                same_as_mv_count += 1
            if ans_status != 0: # 0: only one class, 1: exist different class, 2: only same ans
                multiple_ans_count += 1
            if ans_status == 2:
                multiple_same_class_count += 1
            this_models = get_models_from_baf(self.models_and_scores.all_models, ext_m)
            this_res = np.zeros((len(this_models)))
            model_len += len(this_models)
            msimp_one = 0
            for j, m in enumerate(this_models):
                # accuracy
                this_res[j] = m.predict(x.reshape(1, -1))[0]
                # m simp
                msimp_one += self.models_and_scores.simplicity[ext_m[j]]
            res[i] = np.bincount(this_res.astype(np.int64)).argmax()
            msimp += msimp_one / len(this_models)
        acc = accuracy_score(self.d.y_test.values, res)
        model_len = model_len / test_len
        model_len /= len(self.models_and_scores.all_models)
        msimp /= test_len
        same_as_mv_count /= test_len
        multiple_ans_count /= test_len
        multiple_same_class_count /= test_len
        return [acc, msimp, model_len, model_len, 1.0, 0, multiple_ans_count, multiple_same_class_count, same_as_mv_count]

    def one_cfx_val_ours(self, model_set, x, this_res):
        val = 0
        for i, m in enumerate(model_set):
            this_val = 0
            this_cfx = get_cfx_for_x(m, x=x, original_class=this_res, X_train1=self.d.X_train)
            # get cfx validity
            if this_cfx is None:
                continue
            for m2 in model_set:
                if m2.predict(this_cfx.reshape(1, -1)) != m2.predict(x.reshape(1, -1)):
                    this_val += 1
            this_val /= len(model_set)
            val += this_val
        return val / len(model_set)

    def eval_ours(self):
        self.ours_scores = self.eval_ours_one()
        self.ours_a_scores = self.eval_ours_one(acc_preferred=True)
        self.ours_s_scores = self.eval_ours_one(simp_preferred=True)
        self.ours_as_scores = self.eval_ours_one(as_preferred=True)

    def print_results(self):
        print("se", self.se_scores)
        print("mvc", self.mvc_scores)
        print("mvca", self.mvca_scores)
        print("mvcvv", self.mvcvv_scores)
        print("mvcvva", self.mvcvva_scores)
        print("mvcv", self.mvcv_scores)
        print("mvcva", self.mvcva_scores)
        print("single", self.single_model_avg_acc)


# extract a BAF and get extensions for 1 input x
def get_baf(mas, res, cfxs_pred, input_idx=0, run_id=0, acc_preferred=False, simp_preferred=False, as_preferred=False,
            dataset_name="credit"):
    # build baf:
    fname = f"bafs_{dataset_name}/run{run_id}/bafmm_{input_idx}.af"
    if acc_preferred:
        fname = f"bafs_{dataset_name}/run{run_id}/bafmm_{input_idx}_acc.af"
    if simp_preferred:
        fname = f"bafs_{dataset_name}/run{run_id}/bafmm_{input_idx}_simp.af"
    if as_preferred:
        fname = f"bafs_{dataset_name}/run{run_id}/bafmm_{input_idx}_as.af"
    f = open(fname, 'w')
    # inform solver that this program is a baf and compute s-preferred extension
    f.write("baf.\n")
    f.write("s_prefex.\n")
    # 1) add arg
    for i, m in enumerate(mas.all_models):
        f.write(f"arg(m{i}).\n")
        f.write(f"arg(c{i}).\n")
    # 2) add attacks and supports
    for i, m1 in enumerate(mas.all_models):
        # add support: M x C, C x M
        f.write(f"support(m{i},c{i}).\n")
        f.write(f"support(c{i},m{i}).\n")
        for j, m2 in enumerate(mas.all_models):
            # add attack: M x C, C x M
            if res[j] == cfxs_pred[j][i]:
                if preferred_over(mas, j, i, acc_preferred, simp_preferred, as_preferred):
                    f.write(f"att(m{j},c{i}).\n")
                if preferred_over(mas, i, j, acc_preferred, simp_preferred, as_preferred):
                    f.write(f"att(c{i},m{j}).\n")
            if res[i] != res[j]:
                if j < i:
                    continue
                if preferred_over(mas, j, i, acc_preferred, simp_preferred, as_preferred):
                    f.write(f"att(m{j},m{i}).\n")
                if preferred_over(mas, i, j, acc_preferred, simp_preferred, as_preferred):
                    f.write(f"att(m{i},m{j}).\n")
    f.close()


def preferred_over(mas, i, j, acc_preferred=False, simp_preferred=False, as_preferred=False):
    # if equal, return True
    if not acc_preferred and not simp_preferred and not as_preferred:
        return True
    if acc_preferred:
        if mas.accuracies[i] >= mas.accuracies[j]:
            return True
        else:
            return False
    if simp_preferred:
        if mas.simplicity[i] >= mas.simplicity[j]:
            return True
        else:
            return False
    if as_preferred:
        if mas.accuracies[i] < mas.accuracies[j] and mas.simplicity[i] < mas.simplicity[j]:
            return False
        else:
            return True


def get_extensions_from_baf(input_idx=0, run_id=0, acc_preferred=False, simp_preferred=False, as_preferred=False,
                            dataset_name="credit"):
    """
    need to first execute: clingo bafs/bafmm_i.af baf.dl filter.lp 0 | Out-File -FilePath bafs/output_i.txt
    i is the input index
    """
    ext_m_full = {}
    ext_c_full = {}
    ext_all_full = {}
    m_preferred_idx = 1
    m_re = re.compile("in\(m\d+\)")
    c_re = re.compile("in\(c\d+\)")
    fname = f"bafs_{dataset_name}/run{run_id}/output_{input_idx}.txt"
    if acc_preferred:
        fname = f"bafs_{dataset_name}/run{run_id}/output_{input_idx}_acc.txt"
    if simp_preferred:
        fname = f"bafs_{dataset_name}/run{run_id}/output_{input_idx}_simp.txt"
    if as_preferred:
        fname = f"bafs_{dataset_name}/run{run_id}/output_{input_idx}_as.txt"
    f = open(fname, 'r')
    curr_ans = 0
    to_find = False
    ext_m = []
    ext_c = []
    ext_all = []
    num_ans = 0
    for line in f:
        if line.startswith("Answer:"):
            num_ans += 1
            curr_ans += 1
            ext_m = []
            ext_c = []
            ext_all = []
            to_find = True
            continue
        if to_find:
            ms = m_re.finditer(line)
            for item in ms:
                ext_m.append(int(item[0][4:-1]))
                ext_all.append(item[0][3:-1])
            cs = c_re.finditer(line)
            for item in cs:
                ext_c.append(int(item[0][4:-1]))
                ext_all.append(item[0][3:-1])
            ext_m_full[curr_ans] = ext_m
            ext_c_full[curr_ans] = ext_c
            ext_all_full[curr_ans] = ext_all
            to_find = False
    f.close()
    return ext_m_full, ext_c_full, ext_all_full, num_ans


def get_m_preferred_ext(ext_m_full, ext_c_full, ext_all_full, x, all_models):
    """
    return m, c, and full, and answer status {0, 1, 2} (see below), and whether it is the same as majority vote (bool)
    """
    ans_status = 0  # 0: only one class, 1: exist different class, 2: only same ans
    same_as_mv = False
    lens = np.zeros((len(ext_m_full.keys())))
    for i, key in enumerate(ext_m_full.keys()):
        lens[i] = len(ext_m_full[key])
    idxs = np.where(lens == lens.max())
    # find majority class
    res = np.zeros((len(all_models)))
    for i, m in enumerate(all_models):
        res[i] = m.predict(x.reshape(1, -1))[0]
    predictions, counts = np.unique(res, return_counts=True)
    first_class, _ = predictions[np.argpartition(counts, -1)[-1]], counts[np.argpartition(counts, -1)[-1]]
    # get prediction
    ext_m_first_ans = ext_m_full[idxs[0][0] + 1]
    this_models_first_ans = get_models_from_baf(all_models, ext_m_first_ans)
    first_ans = this_models_first_ans[0].predict(x.reshape(1, -1))[0]
    if len(idxs[0]) == 1:  # idxs[0] is the index array
        if int(first_ans) == int(first_class):
            same_as_mv = True
        return ext_m_full[idxs[0][0] + 1], ext_c_full[idxs[0][0] + 1], ext_all_full[idxs[0][0] + 1], ans_status, same_as_mv
    # HANDLE MULTIPLE M-PREFERRED SETS: return the one which predicts the majority class
    chosen_i = 0
    different_ans_count = 0
    for idx, _ in enumerate(idxs[0]):
        ext_m = ext_m_full[idxs[0][idx] + 1]
        this_models = get_models_from_baf(all_models, ext_m)
        if int(this_models[0].predict(x.reshape(1, -1))[0]) == int(first_class):
            chosen_i = idx
            same_as_mv = True
            #break
        if int(this_models[0].predict(x.reshape(1, -1))[0]) != first_ans:
            different_ans_count += 1
    if different_ans_count == 0:
        ans_status = 2
    else:
        ans_status = 1
    return ext_m_full[idxs[0][chosen_i] + 1], ext_c_full[idxs[0][chosen_i] + 1], ext_all_full[
        idxs[0][chosen_i] + 1], ans_status, same_as_mv


def get_models_cfxs_from_baf(all_models, all_cfxs, ext_m, ext_c):
    return list(np.array(all_models)[ext_m]), list(np.array(all_cfxs)[ext_c])


def get_models_from_baf(all_models, ext_m):
    return list(np.array(all_models)[ext_m])


def check_baf_correctness(all_models, ext_m, ext_c, x, baf_cfxs_pred):
    # check correctness of prediction consistency
    baf_x_pred = all_models[ext_m[0]].predict(x.reshape(1, -1))[0]
    for i in ext_m:
        assert all_models[i].predict(x.reshape(1, -1))[0] == baf_x_pred
    assert len(ext_m) != 0
    assert len(ext_c) != 0
    # check correctness of cfxs robustness
    for i in ext_c:
        for j in ext_m:
            assert baf_cfxs_pred[j][i] != baf_x_pred


class ExperimentsForOneModelSize:
    def __init__(self, d, clfs, accuracy_scores, simplicity_scores, model_size=10, dataset_name="credit", seed=15252):
        self.d = d
        self.clfs = clfs
        self.accuracy_scores = np.array(accuracy_scores)
        self.simplicity_scores = np.array(simplicity_scores)
        self.model_size = model_size
        self.dataset_name = dataset_name
        self.exps = []
        # get clfs for exps
        for i in range(5):
            np.random.seed(seed * (i + 3) + 4321)
            total_num = len(clfs) - 1
            # select models
            m_idxs = np.random.randint(0, total_num, self.model_size)
            all_models = list(np.array(clfs)[m_idxs])
            simplicity_scores0 = np.array(self.accuracy_scores)[m_idxs]
            accuracy_scores0 = np.array(self.simplicity_scores)[m_idxs]
            models_and_scores = ModelsAndScores(all_models, simplicity_scores0, accuracy_scores0)
            self.exps.append(Experiment(self.d, models_and_scores, dataset_name=self.dataset_name, run_id=i))

    def eval(self):
        for i in tqdm(range(5)):
            self.exps[i].eval()

    def eval_ours(self):
        for i in tqdm(range(5)):
            self.exps[i].eval_ours()

    def print_results(self):
        mvca_scores = []
        mvcva_scores = []
        ours_scores = []
        ours_a_scores = []
        ours_s_scores = []
        ours_as_scores = []
        single_model_acc = []
        for i in range(5):
            mvca_scores.append(self.exps[i].mvca_scores)
            mvcva_scores.append(self.exps[i].mvcva_scores)
            ours_scores.append(self.exps[i].ours_scores)
            ours_a_scores.append(self.exps[i].ours_a_scores)
            ours_s_scores.append(self.exps[i].ours_s_scores)
            ours_as_scores.append(self.exps[i].ours_as_scores)
            single_model_acc.append(self.exps[i].single_model_avg_acc) #multiple_ans_count, same_as_mv_count, multiple_same_class_count]
        scores_names = ["name", "acc", "m simp", "m len", "c len", "c val", "abstain", "multi_ans", "multi_same_class", "same_as_mv"]
        scores_table = [scores_names,
                        #np.concatenate((["se"], np.mean(se_scores, axis=0))),
                        #np.concatenate((["mvc"], np.mean(mvc_scores, axis=0))),
                        np.concatenate((["mvca"], np.mean(mvca_scores, axis=0))),
                        #np.concatenate((["mvcv"], np.mean(mvcv_scores, axis=0))),
                        np.concatenate((["mvcva"], np.mean(mvcva_scores, axis=0))),
                        #np.concatenate((["mvcvv"], np.mean(mvcvv_scores, axis=0))),
                        #np.concatenate((["mvcvva"], np.mean(mvcvva_scores, axis=0))),
                        np.concatenate((["ours"], np.mean(ours_scores, axis=0))),
                        np.concatenate((["ours-a"], np.mean(ours_a_scores, axis=0))),
                        np.concatenate((["ours-s"], np.mean(ours_s_scores, axis=0))),
                        np.concatenate((["ours-as"], np.mean(ours_as_scores, axis=0)))]
        scores_table2 = [scores_names,
                         #np.concatenate((["se"], np.std(se_scores, axis=0))),
                         #np.concatenate((["mvc"], np.std(mvc_scores, axis=0))),
                         np.concatenate((["mvca"], np.std(mvca_scores, axis=0))),
                         #np.concatenate((["mvcvv"], np.std(mvcvv_scores, axis=0))),
                         #np.concatenate((["mvcvva"], np.std(mvcvva_scores, axis=0))),
                         #np.concatenate((["mvcv"], np.std(mvcv_scores, axis=0))),
                         np.concatenate((["mvcva"], np.std(mvcva_scores, axis=0))),
                         np.concatenate((["ours"], np.std(ours_scores, axis=0))),
                         np.concatenate((["ours-a"], np.std(ours_a_scores, axis=0))),
                         np.concatenate((["ours-s"], np.std(ours_s_scores, axis=0))),
                         np.concatenate((["ours-as"], np.std(ours_as_scores, axis=0)))]
        print(f"model size = {self.model_size}, mean, single model acc: {np.mean(single_model_acc)}")
        print(tabulate(scores_table, headers="firstrow", tablefmt="outline", floatfmt=".3f"))
        print(f"model size = {self.model_size}, std, single model acc: {np.std(single_model_acc)}")
        print(tabulate(scores_table2, headers="firstrow", tablefmt="outline", floatfmt=".6f"))
