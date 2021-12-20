# Reduce the number of patterns by remove pubchem patterns that are:
# 1. too small (2 atoms)
# 2. overlapping (one pattern is substructure of another pattern)

import pandas as pd
from rdkit import Chem


def main():
    df = pd.read_csv(
        "contextSub/resources/pubchemFPKeys_to_SMARTSpattern.csv", index_col=0
    )
    remove_indices = []
    for i, smarts in enumerate(df["SMARTS"]):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern.GetNumAtoms() == 2:
            remove_indices.append(i)

    i = 0
    while i < len(df):
        pattern = Chem.MolFromSmarts(df["SMARTS"][i])
        j = i + 1
        while j < len(df):
            pattern_2 = Chem.MolFromSmarts(df["SMARTS"][j])
            match = pattern_2.GetSubstructMatch(pattern)
            if len(match) > 0:
                remove_indices.append(i)
                break
            j += 1
        i += 1

    retain_indices = list(set(list(range(len(df)))) - set(remove_indices))
    filtered_df = df.iloc[retain_indices]
    filtered_df.reset_index(drop=True)
    filtered_df.to_csv(
        "contextSub/resources/pubchemFPKeys_to_SMARTSpattern_filtered.csv"
    )


if __name__ == "__main__":
    main()
