import numpy as np
import pandas as pd
# from ROOT import TLorentzVector

# TODO: Think about variables in training!
# MT, DPhill, mll, total pt, DEtajj, met_et, mjj

class VariableConstructor:
    def __init__(self):
        """
        Placeholder
        """

    # def _invMass(self, df):
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
    def DPhill(self, df):
        if not "lep_phi" in list(df.columns):
            raise KeyError("Variable 'lep_phi' not available in inputs!")
        phi1 = np.array([e[0] for e in df["lep_phi"] if len(e) > 0])
        phi2 = np.array([e[1] for e in df["lep_phi"] if len(e) > 1])
        data = abs(phi1 - phi2)
        df = pd.DataFrame(data, columns=["DPhill"])
        return df
    def Mll(self, df):
        # M2 = 2 * pt1 * pt2 * ( cosh( eta1-eta2 ) - cos( phi1-phi2) )
        lepton_pt = np.array(list(map(list, df["lep_pt"])))
        # the following line seems to be necessary for data, don't know whay :(
        lepton_pt = np.array([[l[0], l[1]] for l in lepton_pt])
        lepton_eta = np.array(list(map(list, df["lep_eta"])))
        lepton_eta = np.array([[l[0], l[1]] for l in lepton_eta])
        lepton_phi = np.array(list(map(list, df["lep_phi"])))
        lepton_phi = np.array([[l[0], l[1]] for l in lepton_phi])
        prodPt = np.prod(lepton_pt, axis=1)
        diffEta = -np.diff(lepton_eta, axis=1).T.flatten() 
        diffPhi = -np.diff(lepton_phi, axis=1).T.flatten()
        M2 = 2 * prodPt * (np.cosh( diffEta ) - np.cos( diffPhi))
        M = np.sqrt(M2)
        df = pd.DataFrame(M, columns=["Mll"])
        return df
    def PtTotal(self, df):
        # summed pT of objects
        lepton_pt = np.array(list(map(list, df["lep_pt"])))
        lepton_pt = np.array([[l[0], l[1]] for l in lepton_pt])
        sumPtLep = np.sum(lepton_pt, axis=1)
        jet_pt = np.array(list(map(list, df["jet_pt"])))
        jet_pt = np.array([[l[0], l[1]] for l in jet_pt])
        sumPtJet = np.sum(jet_pt, axis=1)
        data = sumPtLep + sumPtJet
        df = pd.DataFrame(data, columns=["PtTotal"])
        return df
    def DEtajj(self, df):
        if not "jet_eta" in list(df.columns):
            raise KeyError("Variable 'jet_eta' not available in inputs!")
        eta1 = np.array([e[0] for e in df["jet_eta"] if len(e) > 0])
        eta2 = np.array([e[1] for e in df["jet_eta"] if len(e) > 1])
        data = abs(eta1 - eta2)
        df = pd.DataFrame(data, columns=["DEtajj"])
        return df
    def Mjj(self, df):
        # M2 = 2 * pt1 * pt2 * ( cosh( eta1-eta2 ) - cos( phi1-phi2) )
        jet_pt = np.array(list(map(list, df["jet_pt"])))
        jet_eta = np.array(list(map(list, df["jet_eta"])))
        jet_phi = np.array(list(map(list, df["jet_phi"])))
        prodPt = np.prod(jet_pt, axis=1)
        diffEta = -np.diff(jet_eta, axis=1).T.flatten() 
        diffPhi = -np.diff(jet_phi, axis=1).T.flatten()
        M2 = 2 * prodPt * (np.cosh( diffEta ) - np.cos( diffPhi))
        M = np.sqrt(M2)
        df = pd.DataFrame(M, columns=["Mjj"])
        return df
    def MET(self, df):
        pass
        #met_et
    def MT(self, df):
        pass
    
    
def buildVariable(df, name):
    print("building variable {}...".format(name))
    constructor = VariableConstructor()
    func = getattr(constructor, name)
    df_var = func(df)
    return df_var
    
def addVariable(df, name):
    df_var = buildVariable(name, df)
    df = pd.concat([df, df_var], axis=1)
    return df

def addVariables(df, names):
    for n in names:
        df_var = buildVariable(df, n)
        df = pd.concat([df, df_var], axis=1)
    return df

def makeSelections(df, selections = [[]]):
    """
    Make some more selections on dataframe
    """
    for sel in selections:
        if sel:
            df = df[ eval("df['"+sel[0]+"'] "+sel[1]+" "+str(sel[2]))]
    return df

