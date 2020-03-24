import numpy as np
import pandas as pd

# TODO: Think about variables in training!
# MT, DPhill, mll, total pt, DYjj, met_et

class VariableConstructor:
    def __init__(self):
        """
        Placeholder
        """
    def leadLepPt(self, df):
        if not "lep_pt" in list(df.columns):
            raise KeyError("Variable 'lep_pt' not available in inputs!")
        data = np.array([e[0] for e in df["lep_pt"] if len(e) > 0])
        df = pd.DataFrame(data, columns=["leadLepPt"])
        return df
    def subleadLepPt(self, df):
        if not "lep_pt" in list(df.columns):
            raise KeyError("Variable 'lep_pt' not available in inputs!")
        data = np.array([e[1] for e in df["lep_pt"] if len(e) > 1])
        df = pd.DataFrame(data, columns=["subleadLepPt"])
        return df
    def leadJetPt(self, df):
        if not "jet_pt" in list(df.columns):
            raise KeyError("Variable 'jet_pt' not available in inputs!")
        data = np.array([e[0] for e in df["jet_pt"] if len(e) > 0])
        df = pd.DataFrame(data, columns=["leadJetPt"])
        return df
    def subleadJetPt(self, df):
        if not "jet_pt" in list(df.columns):
            raise KeyError("Variable 'jet_pt' not available in inputs!")
        data = np.array([e[1] for e in df["jet_pt"] if len(e) > 1])
        df = pd.DataFrame(data, columns=["subleadJetPt"])
        return df
        
def buildVariable(name, df):
    constructor = VariableConstructor()
    func = getattr(constructor, name)
    df_var = func(df)
    return df_var
    
def addVariable(name, df):
    df_var = buildVariable(name, df)
    df = pd.concat([df, df_var], axis=1)
    return df

def addVariables(names, df):
    for n in names:
        df_var = buildVariable(n, df)
        df = pd.concat([df, df_var], axis=1)
    return df

