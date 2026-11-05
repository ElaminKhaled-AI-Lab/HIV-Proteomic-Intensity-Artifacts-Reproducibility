#!/usr/bin/env python3
"""
R09_druggability_direct_targets.py
Exploratory druggability note (Reviewer 1 comment 8; Reviewer 2 comment 8).

Separates DIRECT pharmacological targets (ChEMBL curated mechanism of action,
direct_interaction = true; queried 2026-10-30 via the ChEMBL API) from
pathway-level associations. For each candidate the table records whether the
direct target was measured on the Olink panel and its association with METS
in the primary (shift-adjusted) arm. Nothing here is evidence of efficacy.
"""
import pandas as pd

import common as C

CHEMBL_MOA = [  # drug, ChEMBL ID, mechanism (ChEMBL), direct target gene(s)
    ("aspirin", "CHEMBL25", "Cyclooxygenase inhibitor (irreversible acetylation)", "PTGS1;PTGS2"),
    ("vorapaxar", "CHEMBL493982", "Proteinase-activated receptor 1 antagonist", "F2R"),
    ("heparin", "CHEMBL1201657", "Antithrombin-III activator", "SERPINC1"),
    ("pentoxifylline", "CHEMBL628", "3',5'-cyclic phosphodiesterase inhibitor; adenosine A2 receptor antagonist",
     "PDE (non-selective);ADORA2A;ADORA2B"),
]
# proteins previously cited as aspirin-related: these are NOT direct aspirin targets
PREVIOUSLY_CITED = {"PLA2G4A": "upstream enzyme of the eicosanoid cascade (releases arachidonic acid); "
                               "not inhibited by aspirin",
                    "F2R": "thrombin receptor PAR-1; direct antagonist is vorapaxar, not aspirin",
                    "GP6": "collagen receptor GPVI; investigational direct inhibitors (e.g. glenzocimab); "
                           "not an aspirin target"}

ids = C.participant_ids()
P = C.olink_matrix(ids)
pm = pd.read_csv(C.TAB / "R05_protein_METS_shift_adjusted.csv").set_index("protein")
rows = []
for drug, cid, moa, targets in CHEMBL_MOA:
    for t in targets.split(";"):
        meas = t in P.columns
        rows.append(dict(drug=drug, chembl_id=cid, chembl_mechanism=moa, direct_target=t,
                         target_on_olink_panel=meas,
                         beta_METS=pm.beta_METS.get(t) if meas else None,
                         q_METS=pm.q.get(t) if meas else None))
tab = pd.DataFrame(rows)
tab.to_csv(C.TAB / "R09_direct_targets.csv", index=False)
pc = pd.DataFrame([dict(protein=k, measured=k in P.columns, relation=v,
                        beta_METS=pm.beta_METS.get(k), q_METS=pm.q.get(k)) for k, v in PREVIOUSLY_CITED.items()])
pc.to_csv(C.TAB / "R09_previously_cited_proteins.csv", index=False)
print(tab.to_string()); print(pc.to_string())
