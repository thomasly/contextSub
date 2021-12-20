Description of manually curated blood-brain-barrier (BBB) compounds dataset files.
dataset files are in .csv format.
•y_test_indices.csv file contains 7162 BBB compound SMILES and permeability of each compound is in binary values (0 for non-permeable and 1 for permeable)
•datasetNormalizedDescrs.csv file contains 1119 2D molecular descriptors of 7162 compounds retrieved from Dragon software.
•y_indices_external.csv file contains 74 compound SMILIES and their permeability class in 0 and 1.    
•external_dataset.csv file contains 2D molecular descriptors of 74 compounds.
Dataset files y_test_indices.csv and datasetNormalizedDescrs.csv used in model training while data in y_indices_external.csv and external_dataset.csv used for external test dataset model validation.
