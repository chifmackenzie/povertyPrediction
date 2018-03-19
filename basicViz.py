# Libraries
import pandas as pd
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set()
# %matplotlib inline

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# load the data sets
loans = pd.read_csv("kiva_loans.csv")
loan_themes = pd.read_csv("loan_theme_ids.csv")

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.rename(index=str, columns={'name': 'country'})
world = world.set_index('country')
world = world.drop(['pop_est', 'gdp_md_est'], axis=1)

loans_per_country = loans['country'].value_counts()
no_countries = []
for country in loans_per_country.index:
    if country not in world.index: no_countries.append(country)

world = world.rename(index={'Dem. Rep. Congo' : 'The Democratic Republic of the Congo', 'Myanmar': 'Myanmar (Burma)',
                   'Lao PDR': "Lao People's Democratic Republic", 'Solomon Is.': 'Solomon Islands',
                   'Dominican Rep.': 'Dominican Republic', 'S. Sudan': 'South Sudan', "Côte d'Ivoire": "Cote D'Ivoire"})
# No Samoa, Saint Vincent and the Grenadines, Virgin Islands, Guam
world['loans_per_country'] = loans_per_country
world.loc[world['loans_per_country'].isnull(), 'loans_per_country'] = 0

ax = world.plot(edgecolor='black', figsize=(14,14), column='loans_per_country', cmap='OrRd', scheme='fisher_jenks')
# ax.set_title('Number of loans by country', fontsize=14)
ax.set_facecolor(sns.color_palette("Blues")[0])
ax.grid(False)
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))

# Data
n_loans_8 = loans['country'].value_counts().sort_values(ascending=False)[:10]
lender_count_mean = loans.groupby('country')['lender_count'].mean().loc[n_loans_8.index]
loan_amount_8 = loans.groupby(['country']).sum()['funded_amount'].sort_values(ascending=False)[:8]
funded_amount_mean = loans.groupby('country')['funded_amount'].mean().loc[loan_amount_8.index]

my_pal = {}
for i, key in enumerate(n_loans_8.keys()):
    my_pal[key] = sns.color_palette()[i%len(sns.color_palette())]
for i, key in enumerate(loan_amount_8.keys()):
    my_pal[key] = sns.color_palette()[i%len(sns.color_palette())]

# First row
sns.barplot(x=n_loans_8, y=n_loans_8.keys(), palette=my_pal, ax=axes[0,0])
sns.barplot(x=lender_count_mean, y=lender_count_mean.keys(), palette=my_pal, ax=axes[0,1])

axes[0,0].set_title("Total number of loans by country", fontsize=14)
axes[0,0].set_ylabel('Country')
axes[0,0].set_xlabel('Number of loans')
axes[0,0].tick_params(axis='y', direction='in', pad=-3, labelsize = 13)
axes[0,0].set_yticklabels(n_loans_8.keys(), horizontalalignment = "left", color="white")
axes[0,0].grid(False)
axes[0,0].set_axisbelow('line')

axes[0,1].set_title('Average number of lenders by country', fontsize=14)
axes[0,1].set_xlabel('Average number of lenders')
axes[0,1].set_ylabel('')
axes[0,1].set_yticklabels(lender_count_mean.keys(), fontsize=12)

# Second row
sns.barplot(x=funded_amount_mean, y=funded_amount_mean.keys(), palette=my_pal, ax=axes[1,1])
sns.barplot(x=loan_amount_8, y=loan_amount_8.keys(), palette=my_pal, ax=axes[1,0])

axes[1,0].set_title("Total funded amount by country", fontsize=14)
axes[1,0].set_ylabel('')
axes[1,0].set_xlabel('Total funded amount')
axes[1,0].tick_params(axis='y', direction='in',pad=-3, labelsize = 13)
axes[1,0].set_yticklabels(loan_amount_8.keys(), horizontalalignment = "left", color="white")
axes[1,0].grid(False)
axes[1,0].set_axisbelow('line')
axes[1,0].set_ylabel('Country')

axes[1,1].set_title('Average funded amount per loan by country', fontsize=14)
axes[1,1].set_xlabel('Average funded amount')
axes[1,1].set_ylabel('')
axes[1,1].set_yticklabels(funded_amount_mean.keys(), fontsize=12)

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(nrows=2, ncols=2)
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1]) 
axes[0,0] = plt.subplot(gs[0])
axes[0,1] = plt.subplot(gs[1])
axes[1,0] = plt.subplot(gs[2])
axes[1,1] = plt.subplot(gs[3])

# Data
activity = loans['activity'].value_counts().sort_values(ascending=False)[:8][::-1]
sector =  loans['sector'].value_counts().sort_values(ascending=False)[:8][::-1]
sector.plot(kind="barh", figsize=(11,8), fontsize = 11, ax=axes[1,0], width=0.65)
distr_list_sec = []
for act_ind in sector.index:
    ar = np.array(loans[(loans['sector'] == act_ind) & (loans['funded_amount'] < 2000)]['funded_amount'])
    distr_list_sec.append(ar)

activity.plot(kind="barh", figsize=(11,8), fontsize = 11, ax=axes[0,0], width=0.65)
distr_list_act = []
for act_ind in activity.index:
    ar = np.array(loans[(loans['activity'] == act_ind) & (loans['funded_amount'] < 2000)]['funded_amount'])
    distr_list_act.append(ar)
    
# First row
axes[0,0].set_title("Number of loans by activity", fontsize=14)
axes[0,0].set_ylabel('Activity')
axes[0,0].set_xlabel('')
axes[0,0].tick_params(axis='y', direction='in',pad=-3, labelsize = 13)
axes[0,0].set_yticklabels(activity.keys(), horizontalalignment = "left", color="white")
axes[0,0].grid(False)
axes[0,0].set_axisbelow('line')

axes[0,1].boxplot(distr_list_act, 0, 'rs', 0, flierprops={'alpha':0.6, 'markersize': 2, 'markeredgecolor': 'None',
                                                          'marker': '.'}, patch_artist=True, medianprops={'color': 'black'})
axes[0,1].set_title('Funded loan distribution\nby activity', fontsize=14)
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel('')
axes[0,1].set_yticklabels(activity.index, fontsize=12)

# Second row
axes[1,0].set_title("Number of loans by sector", fontsize=14)
axes[1,0].set_ylabel('Sector')
axes[1,0].set_xlabel('Number of loans')
axes[1,0].tick_params(axis='y', direction='in',pad=-3, labelsize = 13)
axes[1,0].set_yticklabels(sector.keys(), horizontalalignment = "left", color="white")
axes[1,0].grid(False)
axes[1,0].set_axisbelow('line')

axes[1,1].boxplot(distr_list_sec, 0, 'rs', 0, flierprops={'alpha':0.6, 'markersize': 2, 'markeredgecolor': 'None',
                                                          'marker': '.'}, patch_artist=True, medianprops={'color': 'black'})
axes[1,1].set_title('Funded loan distribution\nby sector', fontsize=14)
axes[1,1].set_ylabel('')
axes[1,1].set_xlabel('Funded loan')
axes[1,1].set_yticklabels(sector.index, fontsize=12)

plt.tight_layout()
plt.show()




















