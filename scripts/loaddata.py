import numpy as np
import pandas as pd
import uproot

def loadData(filepath, variables, selections = [[]],
             nEvents = -1, treename = "mini"):
    """
    opens root file located at 'filepath' (string), converts ROOT's TTree 
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
    returns TTree with name 'treename' in root file at 'filepath'
    """
    f = uproot.open(filepath)
    return f[treename]
    
def _mergeVariableLists(variables, selections):
    """
    helper function to figure out which variables are only used for 
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
