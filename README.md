# friendshoring-in-trade

## Research Question
Have geopolitical alliances evolved into comprehensive strategic partnerships since the election of President Trump, and to what extent has this multidimensional alignment reshaped global trade flows?

## Setup — Prepare Raw Data Files:

### Download Raw Data Files from Zenodo Open Repository (RECOMMENDED)
These datasets have not been modified in anyway, and are the raw datasets from the original sources. All data cleaning and manipulations are done through the pipeline built in data_loader.py. However, since I use datasets from several different sources for my analysis, I put them all in one open repository for easy access and reproducability. 

Steps:
1. Download the raw datasets bundle 'raw.zip' from: https://doi.org/10.5281/zenodo.18112975
2. Extract the ZIP file. You should see a folder named 'raw' inside. 
3. Simply place the 'raw' folder inside the 'data/' directory of this project:

```bash
friendshoring-in-trade/ 
├── README.md    
├── requirements.txt 
├── .gitignore  
├── ...other directories and Files  
├── data/           ← PUT EXTRACTED FOLDER IN HERE   
│   └── README_raw_data.md 
└── results/   
```                     

***OR, if you wish to download the datafiles directly from the original sources, the detailed instructions for doing so are provided in data/README_raw_data.md***

## Setup — Create environment:
```bash

# Windows
python -m venv .venv
.venv\Scripts\activate

# On Mac/Linux:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (to avoid Microsoft Store Python Path Issues):
python -m pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

***EXPECTED OUTPUTS (in results/ directory):***
1. **clusters_2015.csv:**
Results of Hierarchical Clustering (countries grouped into clusters) for 2015.

2. **clusters_2021.csv:**
Results of Hierarchical Clustering (countries grouped into clusters) for 2021.

3. **scree_plot.png:**
Scree Plot depicting the PCA dimensionality reduction, with at what point the 95% variance threshold was reached.

4. **dendrogram_2015.png:**
Dendrogram of clustering results for 2015.

5. **dendrogram_2021.png:**
Dendrogram of clustering results for 2021.

6. **clusters_2015_map.png:**
World Map with countries coloured by their cluster memberships in 2015 for easy comparison.

7. **clusters_2021_map.png:**
World Map with countries coloured by their cluster memberships in 2021 for easy comparison.

8. **regression_table.csv:**
Results of the First-Differences Fixed Effects Regression on the data as a baseline measure of friendshoring for comparison.


## Project Structure

***Expected directory AFTER importing raw data files, BEFORE running main.py:***

```bash

friendshoring-in-trade/
├── README.md                   
├── requirements.txt
├── .gitignore
├── proposal/  
│   ├── PROPOSAL.md          
│   ├── PROPOSAL.pdf          
├── main.py                     
├── src/
│   ├── data_loader.py          
│   ├── models.py               
│   └── evaluation.py           
├── data/
│   ├── README_raw_data.md
│   ├── raw/      
│       ├── AgreementScores.csv
│       ├── cow2iso.csv
│       ├── cultural_distance_PSW2024.csv
│       ├── geodist_PSW24.dta
│       ├── imf_trade_volume(2010-2021).csv
│       ├── linguistic_distance_PSW2024.csv
│       └── religious_distance_PSW2024.dta               
└── results/                # main.py outputs figures & metrics here
```

## Requirements
- Python 3.10.11
- venv
- pandas, numpy, scikit-learn, statsmodels, matplotlib, geopandas