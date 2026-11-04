# Builds the features of the model. The same functions are used for train and test

import numpy as np
import pandas as pd

from src.data import TARGET_YEAR

# Characters of the base 62 counter used in the account IDs
B62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


# Counts how many seasons in a row an account was subscribed, going backwards from last_year
def streak_until(years, last_year):
    years = set(years)
    streak = 0
    while (last_year - streak) in years:
        streak += 1
    return streak


# One row of subscription history features per account
def subscription_features(subs):
    g = subs.groupby('account.id')

    # First we compute the general history features (loyalty and size of the subscriptions)
    feats = pd.DataFrame({
        'sub_n_seasons': g['year'].nunique(),
        'sub_first_year': g['year'].min(),
        'sub_last_year': g['year'].max(),
        'sub_total_tier': g['subscription_tier'].sum(),
        'sub_mean_seats': g['no.seats'].mean(),
        'sub_n_locations': g['location'].nunique(),
        'sub_full_share': g['is_full'].mean(),
        'sub_multiple_any': g['multiple.subs'].apply(lambda s: (s == 'yes').any()).astype(int),
        'sub_streak': g['year'].apply(lambda y: streak_until(y, TARGET_YEAR - 1)),
        'sub_n_last5': g['year'].apply(lambda y: (y >= TARGET_YEAR - 5).sum()),
    })

    # Then we add a flag for each of the last 3 seasons
    for k in [1, 2, 3]:
        season = TARGET_YEAR - k
        feats['sub_{}'.format(season)] = g['year'].apply(lambda y: int(season in set(y)))

    # Finally we describe the most recent subscription of each account
    last = subs.sort_values('year').groupby('account.id').tail(1).set_index('account.id')
    feats['sub_last_seats'] = last['no.seats']
    feats['sub_last_tier'] = last['subscription_tier']
    feats['sub_last_full'] = last['is_full']
    feats['sub_last_price_level'] = last['price.level']

    return feats


# One row of ticket history features per account
def ticket_features(tickets):
    g = tickets.groupby('account.id')

    # We count the tickets and seasons, and check if the account went to a family concert
    feats = pd.DataFrame({
        'tix_n': g.size(),
        'tix_n_seasons': g['year'].nunique(),
        'tix_last_year': g['year'].max(),
        'tix_total_seats': g['no.seats'].sum(),
        'tix_mean_price_level': g['price.level'].mean(),
        'tix_family': g['location'].apply(lambda s: (s == 'Family concert').any()).astype(int),
    })

    # We also add a flag for the accounts that bought tickets in 2013-14
    feats['tix_{}'.format(TARGET_YEAR - 1)] = g['year'].apply(lambda y: int((TARGET_YEAR - 1) in set(y)))

    return feats


# The 37 base features (subscriptions, tickets, donations and location), one row per account
def build_features(ids, data):
    accounts, zipcodes = data['accounts'], data['zipcodes']
    X = pd.DataFrame(index=pd.Index(ids, name='account.id'))

    # First we join the subscription and ticket history. Accounts without history get NaN here
    X = X.join(subscription_features(data['subs'])).join(ticket_features(data['tickets']))
    X['has_sub'] = X['sub_n_seasons'].notna().astype(int)
    X['has_tix'] = X['tix_n'].notna().astype(int)

    # Then we turn the years into recency and tenure, using 30 years when there is no history
    X['years_since_sub'] = (TARGET_YEAR - X['sub_last_year']).fillna(30)
    X['sub_tenure'] = (TARGET_YEAR - X['sub_first_year']).fillna(0)
    X['years_since_tix'] = (TARGET_YEAR - X['tix_last_year']).fillna(30)
    X = X.drop(columns=['sub_last_year', 'sub_first_year', 'tix_last_year'])

    # For the counts no history means 0. For the price levels we use a middle value (2.5)
    count_cols = [c for c in X.columns if c.startswith(('sub_', 'tix_')) and 'price_level' not in c]
    X[count_cols] = X[count_cols].fillna(0)
    X['sub_last_price_level'] = X['sub_last_price_level'].fillna(2.5)
    X['tix_mean_price_level'] = X['tix_mean_price_level'].fillna(2.5)

    # Now we add the donation features. Amounts have very long tails so we use log1p
    acc = accounts.set_index('account.id').loc[ids]
    X['don_2013'] = np.log1p(acc['amount.donated.2013'].values)
    X['don_lifetime'] = np.log1p(acc['amount.donated.lifetime'].values)
    X['don_n'] = acc['no.donations.lifetime'].values
    first_year = acc['first.donated'].dt.year
    X['years_since_first_don'] = (TARGET_YEAR - first_year).fillna(0).values
    X['first_don_2014'] = (first_year == TARGET_YEAR).astype(int).values
    X['has_relationship'] = acc['relationship'].notna().astype(int).values
    X['has_billing_zip'] = acc['billing.zip.code'].notna().astype(int).values

    # After this we add the location features by joining the zip code table
    z = acc[['zip5']].merge(zipcodes[['zip5', 'State', 'Lat', 'Long', 'EstimatedPopulation', 'avg_income']],
                            on='zip5', how='left')
    X['in_CA'] = (z['State'] == 'CA').astype(int).values

    # Distance to San Francisco with the haversine formula. Unknown locations get a far distance (5000 km)
    lat1, lon1 = np.radians(37.7749), np.radians(-122.4194)
    lat2, lon2 = np.radians(z['Lat'].values), np.radians(z['Long'].values)
    a = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))
    X['dist_sf_km'] = np.log1p(np.nan_to_num(dist, nan=5000))

    # Finally, the population and income of the zip code. Missing values are filled with the median
    X['zip_log_pop'] = np.log1p(z['EstimatedPopulation'].values)
    X['zip_log_income'] = np.log1p(z['avg_income'].values)
    X['zip_log_pop'] = X['zip_log_pop'].fillna(X['zip_log_pop'].median())
    X['zip_log_income'] = X['zip_log_income'].fillna(X['zip_log_income'].median())

    return X


# Converts the counter part of a Salesforce ID (characters 5 to 14) from base 62 to a number
def id_number(account_id):
    number = 0
    for ch in account_id[5:15]:
        number = number * 62 + B62.index(ch)
    return number


# Account age: position of each account in the creation order of all the accounts (0 = oldest, 1 = newest)
def id_rank(ids, accounts):
    # We rank against all the accounts in account.csv, so train and test use the same reference
    all_numbers = np.sort(accounts['account.id'].map(id_number).values)
    return np.searchsorted(all_numbers, ids.map(id_number).values, side='right') / len(all_numbers)


# Features of the final model: the 37 base features plus the account age
def build_final_features(ids, data):
    return build_features(ids, data).assign(id_rank=id_rank(ids, data['accounts']))
