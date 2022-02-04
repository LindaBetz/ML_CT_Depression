# Machine Learning Approach to the Role of Childhood Trauma on Adult Depressive Affect

R-code to reproduce analyses described in "The role of childhood abuse and neglect in depressive affect in adulthood: a machine learning approach in a general population sample" by Linda T. Betz, Marlene Rosen, Raimo Salokangas, and Joseph Kambeitz.

Code by L. Betz (linda.betz@uk-koeln.de)

There are two files:

* Main_Analysis.R provides code to reproduce results and plots reported in the main manuscript.
* Custom_Functions.R provides several custom functions used in Code_Main_Analysis.R.

Data required for the analysis (MIDUS samples) are available for public use at https://www.icpsr.umich.edu/web/pages/:

* MIDUS Biomarker: https://doi.org/10.3886/ICPSR29282.v9

* MIDUS Refresher: https://doi.org/10.3886/ICPSR36532.v3

* MIDUS II: https://doi.org/10.3886/ICPSR04652.v7

* MIDUS Biomarker Refresher: https://doi.org/10.3886/ICPSR36901.v6


For optimal reproducibility, we use the R-package "checkpoint", with snapshot date June 21, 2021 (R version 4.1.0). Information on how this works can be found at https://mran.microsoft.com/documents/rro/reproducibility.
