# Friendshoring and Trade: Analysing Trade Flow Shifts in the Post-Trump Era

## Motivation
The term “friendshoring” has become a buzzword in trade discussions, especially amidst the 
broad-spectrum tariffs introduced by the United States. It refers to the phenomenon of global 
trade flows being dictated increasingly by geopolitical considerations rather than economic 
pragmatism, opposing traditional views in favour of free trade and globalisation. A study 
conducted by the McKinsey Global Institute finds early evidence of friendshoring, pointing at 
the United States’ active efforts to reduce the “geopolitical distance” covered by its trade by 10% 
since 2017 — just one year after the election of Donald Trump. The use of trade policy as a 
geopolitical tool has since become increasingly common, exacerbated by the U.S.-China Rivalry, 
the Russo-Ukrainian War, the rush to secure supply chains after the COVID-19 Pandemic, and 
numerous other such frictions. 

## Research Question  
This project aims to investigate to what extent have global trade flows been reorganised along 
geopolitical lines since the election of President Trump. If more of global trade can be explained 
by geopolitical alignment, that indicates the presence of friendshoring behaviour among nations. 
Cultural and linguistic similarities as well as geographic distance will also be included as 
attributes of each dyadic country-pair so that the effect of geopolitical interests in particular can 
be further isolated.

## Methodology
Dyadic (country-pair) datasets for trade volume, geopolitical distance, cultural distance, religious 
distance, linguistic distance, and geographic distance will be transformed into 6 country-level 
data matrices — one for each attribute variable. These will be square matrices, with the rows and 
columns both being all the countries in the dataset. The cell entries will be the respective values 
of each attribute variable for the dyadic relationship between those two countries. This will be 
done for 2015, which is the year before President Trump first entered office, and 2021, which is 
one year after the outbreak of COVID-19. This also avoids potential confounding factors from 
the Russo-Ukrainian War, which started in 2022.  
The six attribute matrices will be combined into a single country-level feature set, with 
dimensionality reduction (PCA) applied to manage the high-dimensional space before clustering. 
This allows for the application of a hierarchical clustering algorithm separately on the 2015 and 
2021 datasets, clustering countries based on the similarities of their attributes across all country
level matrices without any prior knowledge of real-world alliances and groupings. The resulting 
clusters for each year will then be analysed and compared to real-world geopolitical groupings 
(NATO, BRICS, QUAD, SCO, etc…). If the algorithm’s clustering for 2021 reflects real-world 
alliances more closely as compared to the clustering for 2015, that will serve as evidence of the 
presence of friendshoring.  
While the methodology would benefit from conducting the clustering for several years, the lack 
of availability of data for all variables makes this impossible. However, the timeliness and choice 
of the years in question mean that significant results may yet be yielded. 

## Data Sources
Trade Volume: IMF and Correlates of War (COW) 
o Will ideally be defined as squared deviations in trade volume from the differential 
trend established by average trade volume for each dyadic pair from 2010 to 2014.
o Disclaimer: Initial survey of data availability suggests this is possible, but I may 
have to change it to percentage change in trade volume or log of trade volume if 
there are too many zero entries. 
• Geopolitical Distance: Ideal Point Data based on UN General Assembly voting patterns 
constructed by Baily, Strezhnev, and Voeten (2017) 
• Cultural Distance (over time), Religious Distance (over time), Linguistic Distance 
(cross-sectional), Geographic Distance (cross-sectional): Database constructed by 
Spolaore and Wacziarg (2025)