# Friendshoring and Trade: Analysing Trade Flow Shifts in the Post-Trump Era and the Relevance of Geopolitics in International Trade

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
This project aims to investigate through a machine learning approach whether geopolitical 
alliances have evolved into comprehensive strategic partnerships since the election of President 
Trump, and to what extent this multidimensional alignment has reshaped global trade flows. 

## Methodology
Dyadic (country-pair) data on trade volume, agreement scores, cultural distance, religious 
distance, linguistic distance, and geographic distance will be transformed into 6 country-pair 
level data matrices — one for each attribute variable. This will be done for 2015, which is the 
year before President Trump first entered office, and 2021, which is one year after the COVID-
19 outbreak. This also avoids potential confounding factors from the Russo-Ukrainian War, 
which started in 2022.  
The six attribute matrices will be combined into a two feature matrices — one for 2015, and one 
for 2021. PCA will then be fitted to a vertical stack of these two feature matrices to ensure that 
the principal components are consistent across both years, thereby ensuring comparable 
clustering results. This fitted PCA will then be applied to the feature matrices for both years, 
after which hierarchical clustering will be conducted, producing the optimal number of clusters k 
as determined by the number that yields the highest silhouette score within the range of (2,10). 
The resulting clusters for each year will then be analysed and compared to real-world 
geopolitical groupings. If clusters in 2021 more closely align with geopolitical blocs like NATO 
or BRICS compared to 2015, this would indicate strengthening friendshoring patterns. 
  
Finally, a simple first-difference fixed effects regression is conducted as a baseline measure of 
the extent to which friendshoring has affected trade volume in particular. The estimated model is 
specified below:

$$
\Delta \log (Trade_{ij}) = \beta_0 + \beta_1 \Delta Friend_{ij} + \beta_2 \Delta Cultural_{ij} + \beta_3 \Delta Religious_{ij} + \epsilon'_{ij}
$$

- **$\Delta \log (Trade_{ij})$**: log change in trade volume in dyad formed by countries $i$ and $j$ from 2015 to 2021  
- **$\Delta Friend_{ij}$**: change in geopolitical distance in dyad formed by countries $i$ and $j$ from 2015 to 2021  
- **$\Delta Cultural_{ij}$**: change in cultural distance in dyad formed by countries $i$ and $j$ from 2015 to 2021  
- **$\Delta Religious_{ij}$**: change in religious distance in dyad formed by countries $i$ and $j$ from 2015 to 2021  
- **$\epsilon'_{ij}$**: error term varied by dyads

## Data Sources
• Trade Volume: Calculated from exports data (free on board, USD) from the IMF 
International Merchandise Trade Statistics (IMTS) Database 
o log(1+x) of squared deviations from differential trend established by average 
trade volume for each dyad from 2010 to 2014.  
• Geopolitical Distance: Calculated from agreement score data constructed by Bailey, 
Strezhnev, and Voeten (2017). 
• Cultural Distance (over time), Religious Distance (over time), Linguistic Distance 
(cross-sectional), Geographic Distance (cross-sectional): Data repository constructed 
by Pellegrino, Spolaore, and Wacziarg (2025).  