import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv('1730285881-Airbnb_Open_Data.csv', low_memory=False)

# Strip spaces in column names
df.columns = df.columns.str.strip()

print("----- First 5 Rows -----")
print(df.head())

print("----- Dataset Info -----")
print(df.info())

print("----- Duplicate Count -----")
print(df.duplicated().value_counts())

# -----------------------------
# Data Cleaning
# -----------------------------

# Fix misspelling check
print("Rows with 'Brookln':")
print(df[df['neighbourhood group'] == 'Brookln'])

# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop columns with insufficient data
df.drop(['house_rules', 'license'], axis=1, inplace=True)

# Clean price and service fee
df['price'] = df['price'].astype(str)
df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df['service fee'] = df['service fee'].astype(str)
df['service fee'] = df['service fee'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df['service fee'] = pd.to_numeric(df['service fee'], errors='coerce')

# Rename columns
df.rename(columns={'price': 'price_$', 'service fee': 'service_fee_$'}, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Convert datatypes
df['id'] = df['id'].astype(str)
df['host id'] = df['host id'].astype(str)
df['last review'] = pd.to_datetime(df['last review'], errors='coerce')
df['Construction year'] = pd.to_numeric(df['Construction year'], errors='coerce').astype('Int64')

# Fix spelling
df.loc[df['neighbourhood group'] == 'Brookln', 'neighbourhood group'] = 'Brooklyn'
df.loc[df['neighbourhood group'] == 'brookln', 'neighbourhood group'] = 'Brooklyn'

# Remove outliers in availability_365
df = df[df['availability 365'] <= 500]

print("----- Cleaned Data Info -----")
print(df.info())
print(df.describe())

# -----------------------------
# Analysis and Plots
# -----------------------------

# Property types
property_types = df['room type'].value_counts().to_frame(name="count")
print("Property Types:\n", property_types)

room_type_bar = plt.bar(property_types.index, property_types["count"])
plt.bar_label(room_type_bar, labels=property_types["count"], padding=4)
plt.ylim([0, 50000])
plt.xlabel('Room type')
plt.ylabel('Count')
plt.title('Room types count')
plt.show()

# Neighborhood group
hood_group = df['neighbourhood group'].value_counts().to_frame(name="count")
print("Neighborhood Group:\n", hood_group)

hood_group_bar = plt.bar(hood_group.index, hood_group["count"])
plt.bar_label(hood_group_bar, labels=hood_group["count"], padding=4)
plt.ylim([0, 40000])
plt.xlabel('Neighborhood group')
plt.ylabel('Listings count')
plt.xticks(rotation=45)
plt.title('Neighborhood group listing counts')
plt.show()

# Average price by neighborhood group
avg_price = df.groupby('neighbourhood group')['price_$'].mean().sort_values(ascending=False).to_frame()
plt.figure()
avg_price_bar = plt.bar(avg_price.index, avg_price['price_$'])
plt.bar_label(avg_price_bar, labels=round(avg_price['price_$'], 2), label_type='edge', padding=4)
plt.ylim([0, 700])
plt.xlabel('Neighborhood group')
plt.ylabel('Average price ($)')
plt.xticks(rotation=45)
plt.title('Average price per listing by Neighborhood group')
plt.show()

# Construction year vs price
plt.figure()
df.groupby('Construction year')['price_$'].mean().plot()
plt.xlabel('Construction year')
plt.ylabel('Average price ($)')
plt.title('Price vs Construction year')
plt.show()

# Top 10 hosts
hosts = df.groupby('host name')['calculated host listings count'].sum().sort_values(ascending=False).nlargest(10).to_frame()
print("Top Hosts:\n", hosts)

hosts_bar = plt.bar(hosts.index, hosts['calculated host listings count'])
plt.bar_label(hosts_bar, labels=hosts['calculated host listings count'], label_type='edge', padding=3)
plt.xlabel('Host Name')
plt.ylabel('Listing count')
plt.xticks(rotation=80)
plt.ylim([0, 120000])
plt.title('Top 10 Hosts by Listing Count')
plt.show()

# Verified host identity vs reviews
review = df.groupby('host_identity_verified')['review rate number'].mean().sort_values(ascending=False).to_frame()
print("Review by Host Identity:\n", review)

review_bar = plt.bar(review.index, review['review rate number'])
plt.bar_label(review_bar, labels=round(review['review rate number'], 2), padding=4)
plt.ylim([0, 5])
plt.xlabel('Host verification')
plt.ylabel('Avg Review Rate')
plt.title('Reviews by Verification Status')
plt.show()

# Correlation: price vs service fee
print("Correlation price vs service fee:", df['price_$'].corr(df['service_fee_$']))

sns.regplot(x='price_$', y='service_fee_$', data=df)
plt.xlabel('Price ($)')
plt.ylabel('Service Fee ($)')
plt.title('Price vs Service Fee')
plt.show()

# Average review rate per neighborhood group and room type
ARRN = df.groupby(['neighbourhood group', 'room type'])['review rate number'].mean().to_frame()
print("Average Review Rate:\n", ARRN)

plt.figure(figsize=[12, 10])
sns.barplot(x='neighbourhood group', y='review rate number', hue='room type', data=df)
plt.xlabel('Neighborhood group')
plt.ylabel('Avg Review rate')
plt.title('Review rate by Neighborhood & Room type')
plt.show()

# Host listing count vs availability
plt.figure()
sns.regplot(x='calculated host listings count', y='availability 365', data=df)
plt.xlabel('Host Listings Count')
plt.ylabel('Availability (365)')
plt.title('Host Listings vs Availability')
plt.show()

print("Correlation host listings vs availability:", df['calculated host listings count'].corr(df['availability 365']))
