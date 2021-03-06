import numpy as np
import pandas as pd
import uproot
import configparser
from variables import *

def getDataInDict(cfg, section = "train"):
    """
    Retrieve training data in dictionary 
    """
    dset = {}
    if "filepathData" in cfg.keys():
        filepath = cfg["filepathData"]
        print("INFO: Loading data for process from filepath '{}' ".format(cfg["filepathData"]))
        df = loadData(filepath, cfg["variables"], selections = cfg["preselections"],
                      nEvents = cfg["nEvents"])
        df = addVariables(df, cfg["variables_to_build"])
        df = makeSelections(df, cfg["selections"])
        dset["Data"] = df
    else:
        for processName, filepath in cfg["samplePaths"].items():
            if filepath:
                print("INFO: Loading data for process '{}' from filepath '{}' " \
                      "".format(processName,filepath))
                df = loadData(filepath, cfg["variables"], selections = cfg["preselections"],
                              nEvents = cfg["nEvents"])
                df = addVariables(df, cfg["variables_to_build"])
                df = makeSelections(df, cfg["selections"])
                dset[processName] = df

    return dset

def readCfg(configPath, section = "train"):
    """
    Read config file into dictionary (customized)
    """
    cfg = {}
    config = configparser.ConfigParser()
    config.read("config.cfg")
    cfg["section"] = section
    cfg["nEvents"] =  int(config[section]["nEvents"])
    samplePaths = {}
    cfg["samples"] = config["common"]["samples"].split(",")
    for s in cfg["samples"]:
        samplePaths[s] = config["common"]["filepath{}".format(s)]
    cfg["samplePaths"] = samplePaths
    cfg["variables"] = config["common"]["variables"].split(",")
    cfg["variables_to_build"] = config["common"]["variables_to_build"].split(",")
    cfg["training_variables"] = config["common"]["training_variables"].split(",")
    cfg["preselections"] = [i[1].split(",") for i in config["preselections"].items()]
    cfg["selections"] = [i[1].split(",") for i in config["selections"].items()]
    cfg["standardize"] = config[section]["standardize"].lower() in ["true", "1", "y"]
    if section == "train":
        cfg["nEpochs"] = config[section]["nEpochs"]
    if section == "analyze":
        cfg["modelPath"] = config[section]["modelPath"]
        cfg["scalerPath"] = config[section]["scalerPath"]
        cfg["referenceMode"] = config[section]["referenceMode"]
        if "filepathData" in config["analyze"].keys():
            cfg["filepathData"] = config[section]["filepathData"]
    return cfg

def loadData(filepath, variables, selections = [[]],
             nEvents = -1, treename = "mini"):
    """
    Open root file located at 'filepath' (string), converts ROOT's TTree 
    format to a pandas dataframe including all 'variables' 
    (list of strings), and returns it.
    Selections can be applied by providing a list 'selections'  in the 
    format: [ [<varname>, <math condition>, <value>], ...]
    """

    # find out variables that are only used for selections
    all_vars, vars_for_selection_only = _mergeVariableLists(variables, selections)

    # load data into dataframe
    tree = _getTTree(filepath, treename)
    data = tree.arrays(branches=all_vars, \
                       entrystop=nEvents, \
                       namedecode = "ascii")
    df = pd.DataFrame(data)

    # make selections
    for sel in selections:
        if sel:
            df = df[ eval("df['"+sel[0]+"'] "+sel[1]+" "+str(sel[2]))]

    # remove variables that were only used for selecting data
    for var in vars_for_selection_only:
        df = df.drop(var, axis=1)

    # reset indexing (and drop old ones)
    # after some events were removed
    df = df.reset_index(drop=True)
    
    return df

def _getTTree(filepath, treename):
    """
    Return TTree with name 'treename' in root file at 'filepath'
    """
    f = uproot.open(filepath)
    return f[treename]
    
def _mergeVariableLists(variables, selections):
    """
    Helper function to figure out which variables are only used for 
    making selections
    """
    vars_for_selection = [e[0] for e in selections if len(e) > 0]
    all_vars = variables
    for var in vars_for_selection:
        if var in variables:
            vars_for_selection.remove(var)
        else: all_vars.append(var)
    # does not preserve order
    all_vars = list(set(all_vars))
    return all_vars, vars_for_selection
