# #Molecular-Bioactivity-Analysis-for-Toxoplasma-gondii
This repository contains a Python script for performing a comprehensive analysis of the molecular bioactivity of compounds targeting Toxoplasma gondii.

Molecular Bioactivity Analysis for Toxoplasma gondii
This repository contains a Python script to perform a comprehensive molecular bioactivity analysis of compounds targeting Toxoplasma gondii. The analysis includes the calculation of molecular descriptors (such as Lipinski's Rule of Five), normalization of activity values to pIC50, data visualization, and statistical analysis using the Mann-Whitney U test.

Structure and Requirements
The main script (toxoplasma_analysis.py) organizes the process into sections for library import, descriptor calculation, data transformation, visualization, and statistical analysis.

Dependencies: You will need Python ≥ 3.7, RDKit, pandas, numpy, matplotlib, seaborn, and scipy. You can install most with pip and RDKit with conda.

Input File: The script expects a CSV file named converted_file.csv with columns for canonical_smiles, standard_value (biological activity value in nM), and bioactivity_class.

How to Run and Results
To run the full analysis, simply run python toxoplasma_analysis.py.

The script will generate:

Curated CSV Files: Multiple datasets with the processed information.

PDF Visualizations: Plots showing bioactivity distribution, relationships between molecular properties and activity, and other visual analyses.

Statistical Outputs: CSV files with the results of the Mann-Whitney U test for each descriptor.

Expected outputs include bioactivity class distribution plots, relationships between molecular properties and activity, and statistical evidence of the influence of descriptors on biological activity.

Important Notes:

IC50 values are limited to a configurable maximum (default 100,000,000 nM).

Conversion to pIC50 is performed using the formula: pIC50 = -log10(IC50 * 10
−9
).
