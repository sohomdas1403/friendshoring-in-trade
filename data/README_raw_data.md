# Download Raw Data Files Directly from Original Sources (non-Zenodo Option)

***NOTE: It is recommended to download all raw data files at once (and properly named) from the Zenodo open repository as instructed in README.md***


Directly downloading the raw datasets from their original sources involves the following:

### Download Agreement Scores Dataset:

1. Download 'AgreementScores.csv' from Erik Voeten's Dataverse in Harvard Dataverse, linked here: https://dataverse.harvard.edu/file.xhtml?fileId=11837234&version=37.0. 
2. Click the blue "Access File" button on the right side, then under "Download Options, click on "Comma Separated Values". 

### Download Cultural Distance, Geographic Distance, Linguistic Distance, and Religious Distance Datasets:
Here, we download 4 datasets from "The Geo-Political Distance Data Repository" constructed by Pellegrino, Spolaore, and Wacziarg for their research article titled "Barriers to Global Capital Allocation" at https://www.geopoliticaldistance.org/home. The links in the step below take you to different subpages all part of the geopoliticaldistance.org webpage for easy following of the steps:

1. Go to https://www.geopoliticaldistance.org/cultural-distance, click on "Comma Separated Values (.csv)", and download the file. This is the Cultural Distance Dataset.
2. Go to https://www.geopoliticaldistance.org/linguistic-distance, click on "Comma Separated Values (.csv)", and download the file. This is the Linguistic Distance Dataset.
3. Go to https://www.geopoliticaldistance.org/geographic-distance, click on "Stata (.dta)", and download the file. This is the Geographic Distance Dataset.
4. Go to https://www.geopoliticaldistance.org/religious-distance, click on "Stata (.dta)", and download the file. This is the Religious Distance Dataset.

### Download the Country Codes Dataset (for Conversion of Country Codes):
1. Go to the publicly available github repository at https://github.com/leops95/cow2iso, and download the csv file titled "cow2iso.csv".

### Download the IMF Trade Volume Dataset from 2010 to 2021:
Here, using the API requires authorisation, and downloading the full dataset in its entirety is inefficiently taxing due to its sheer size (includes classifications by countries, regions, and economic development, dyadic, 1948 to 2025, four indicators per country, at annual, monthly, and quarterly frequencies). Thus, manual download via the data explorer is necessary.

1. Go to IMF's International Trade in Goods (by partner country) (IMTS) Data at https://data.imf.org/en/datasets/IMF.STA:IMTS, and click on "View Data" to open the Data Explorer.
2. On the left-hand side under "Dataset", ensure "International Trade in Goods (by partner country) (IMTS)" is selected.
3. Under "Country", first select "All", which selects all countries (including historic) AND groupings by regions AND groupings by economic development. Then, go through the list and deselect the following: 
    1. Africa
    2. CIS
    3. Europe
    4. Middle East
    5. Middle East, North Africa, Afghanistan, and Pakistan
    6. Other Countries n.i.e
    7. Advanced Economies
    8. EMDEs by Source of Export Earnings: Fuel
    9. EMDEs by Source of Export Earnings: Nonfuel
    10. Emerging and Developing Asia
    11. Emerging and Developing Europe
    12. Emerging Markets and Developing Economies
    13. European Union
    14. Latin America and Caribbean (LAC)
    15. Middle East and Central Asia
    16. Sub-Saharan Africa (SSA)
    17. World
    18. Euro Area (EA)
4. Under "Indicator" select "Exports of goods, Free on board (FOB), US dollar".
5. Under "Counterpart Country", first select "All", which selects all countries (including historic) AND groupings by regions AND groupings by economic development. Then, go through the list and deselect the following: 
    1. Africa not specified
    2. Asia not specified
    3. Belgium-Luxembourg
    4. Countries & Areas not specified
    5. Europe not specified
    6. Middle East and Central Asia not specified
    7. Special Categories
    8. Western Hemisphere not specified
    9. Africa
    10. CIS
    11. Europe
    12. Middle East
    13. Middle East, North Africa, Afghanistan, and Pakistan
    14. Other Countries n.i.e
    15. Taiwan Province of China
    16. Czechoslovakia
    17. Netherlands Antilles
    18. Serbia and Montenegro
    19. Yugoslavia, Socialist Federal Republic of
    20. Advanced Economies
    21. EMDEs by Source of Export Earnings: Fuel
    22. EMDEs by Source of Export Earnings: Nonfuel
    23. Emerging and Developing Asia
    24. Emerging and Developing Europe
    25. Emerging Markets and Developing Economies
    26. European Union
    27. Latin America and Caribbean (LAC)
    28. Middle East and Central Asia
    29. Sub-Saharan Africa (SSA)
    30. World
    31. Euro Area (EA)
6. You should now have 209 countries selected in both "Country" and "Counterpart Country".
7. Under "Frequency", select "Annual".
8. Under "Time Period", choose "Custom" and input the following range: 01/01/2010 — 12/31/2021.
9. Next, click "Apply" at the bottom-right of the selection panel. The Series Count should show 34,091.
10. Click the blue "Download" button on the right-hand side of the website (next to the "Add to Watchlist" button). Then, select "Data on this page".
11. Here, before downloading, select the following:
    1. File Format: CSV
    2. Data format: Observation per row
    3. Metadata: ID and Name
12. Click the blue "Download" button on the bottom-right. After downloading, rename the file to 'imf_trade_volume(2010-2021).csv'. 

### Ready the 'raw' File for Use

1. Place the downloaded datasets into a folder, and name the folder 'raw'. 
2. Simply place the 'raw' folder inside the 'data/' directory of this project:

```bash
friendshoring-in-trade/ 
├── README.md    
├── requirements.txt 
├── .gitignore  
├── ...other directories and Files  
├── data/        ← PUT 'raw' FOLDER IN HERE   
│   └── README_raw_data.md 
└── results/         
```                     