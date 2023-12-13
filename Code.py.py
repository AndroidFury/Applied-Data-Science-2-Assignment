import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 
import seaborn as sns 

 
def transpose_data(file_path: str):
    """
    Reads data from a CSV file, transposes it for analysis, and separates the data into two transposed DataFrames.
    The first DataFrame is transposed by country, with countries as columns.
    The second DataFrame is transposed by year, with years as columns.

    Parameters:
    - file_path: A string path to the CSV file.

    Returns:
    - original_df: The original, unmodified DataFrame.
    - transposed_by_country: A DataFrame with data transposed by country.
    - transposed_by_year: A DataFrame with data transposed by year.
    """
    # Reading the data from the file
    original_df = pd.read_csv(file_path)

    # Transposing the DataFrame
    transposed_df = original_df.transpose()

    # Setting the first row as column names
    transposed_df.columns = transposed_df.iloc[1]

    # Dropping the 'year' row and creating a DataFrame transposed by year
    transposed_by_year = transposed_df[1:].drop('year')

    # Resetting columns to the first row for transposing by country
    transposed_df.columns = transposed_df.iloc[0]

    # Dropping the 'country' row and creating a DataFrame transposed by country
    transposed_by_country = transposed_df[1:].drop('country')

    return original_df, transposed_by_country, transposed_by_year


data = pd.read_csv('cData.csv')

# Display the first 5 rows of the original DataFrame
print("### Original DataFrame")
print(data.head(7))

# Show statistics of Original Data
print("\n### Statistics of Original Data")
print(data.describe())

# Transpose data by country and by year
data_by_country = data.transpose() # Adjust as needed
data_by_year = data.transpose()    # Adjust as needed

# Display transposed data
print("\n### Data Transposed by Country")
print(data_by_country.head(7))
print("\n### Data Transposed by Year")
print(data_by_year)

# Filter for forest data and drop missing values
forest_data = data[['country', 'year', 'forest_area']].dropna()

# Function to filter data by year
def filter_by_year(year):
    """
   Filters the forest data DataFrame for a specific year.

   Parameters:
   - year: An integer value of the year to filter by.

   Returns:
   - A filtered DataFrame containing data only for the specified year.
   """
    return forest_data[forest_data['year'] == year]

# Filter forest data for specific years
forest_data_1990 = filter_by_year(1990)
forest_data_2000 = filter_by_year(2000)
forest_data_2010 = filter_by_year(2010)
forest_data_2012 = filter_by_year(2012)

forest_data_2015 = filter_by_year(2015)
forest_data_2018= filter_by_year(2018)

forest_data_2020 = filter_by_year(2020)

# Plotting
style.use('ggplot')
plt.figure(figsize=(20, 15))
barWidth = 0.1
years = [forest_data_1990, forest_data_2000, forest_data_2010,forest_data_2012, forest_data_2015,forest_data_2018, forest_data_2020]
colors = ['lime', 'blue', 'peru', 'yellow', 'darkviolet','red','green']
labels = ['1990', '2000', '2010','2012', '2015','2018', '2020']

for i, year_data in enumerate(years):
    plt.bar(np.arange(year_data.shape[0]) + i * 0.2, year_data['forest_area'], color=colors[i], width=barWidth, label=labels[i])

plt.legend()
plt.xlabel('Country', fontsize=15)
plt.title("Forest Area by Year", fontsize=15)
countries = forest_data['country'].unique()
plt.xticks(np.arange(len(countries)) + 0.2, countries, fontsize=10, rotation=45)
plt.show()

# Display unique countries
print("\n### Unique Countries in Forest Data")
print(forest_data['country'].unique())

# Display original DataFrame columns
print("\n### Columns in Original DataFrame")
print(data.columns)
    
def get_agricultural_data_for_year(df, year):
    """
  Filters the agricultural land data DataFrame for a specific year.

  Parameters:
  - df: The DataFrame containing agricultural land data.
  - year: An integer value of the year to filter by.

  Returns:
  - A filtered DataFrame containing agricultural land data only for the specified year.
  """
    return df[df['year'] == year]

# Selecting and cleaning agricultural land data
agri_land_data = data[['country', 'year', 'agricultural_land']].dropna()

# Updated years to be analyzed
years = [1990, 1995, 2000, 2005, 2010, 2012, 2015, 2018, 2019]
colors = ['lime', 'magenta', 'blue', 'yellow', 'peru', 'cyan', 'red', 'green', 'orange']
labels = [str(year) for year in years]

# Filter agricultural land data for specified years
agri_land_by_year = {year: get_agricultural_data_for_year(agri_land_data, year) for year in years}

# Plotting
style.use('ggplot')
plt.figure(figsize=(15, 10))
barWidth = 0.1

for i, year in enumerate(years):
    year_data = agri_land_by_year[year]
    plt.bar(np.arange(year_data.shape[0]) + i * 0.1, year_data['agricultural_land'], color=colors[i], width=barWidth, label=str(year))

plt.legend()
plt.xlabel('Country', fontsize=15)
plt.title("Agricultural Land by Year", fontsize=15)
countries = agri_land_data['country'].unique()
plt.xticks(np.arange(len(countries)) + 0.2, countries, fontsize=10, rotation=45)
plt.show()

#null function for set feature
def remove_null_values(feature):
    """
  Removes null values from a feature column in a DataFrame.

  Parameters:
  - feature: A pandas Series from which to drop null values.

  Returns:
  - A NumPy array of the feature with null values removed.
  """
    return np.array(feature.dropna())

# Filter data for Australia
United_data = data[data['country'] == 'United Kingdom']

# Function to clean data by removing null values
def clean_data(dataframe, column_name):
    """
  Cleans a specific column in a DataFrame by removing null values.

  Parameters:
  - dataframe: The DataFrame containing the data to be cleaned.
  - column_name: A string name of the column from which to remove null values.

  Returns:
  - A pandas Series with null values removed from the specified column.
  """
    return dataframe[column_name].dropna().values

# Cleaning data for different features
electricity_access_aus = clean_data(United_data, 'access_to_electricity')
agriculture_land_aus = clean_data(United_data, 'agricultural_land')
co2_emissions_aus = clean_data(United_data, 'co2_emission')
arable_area_aus = clean_data(United_data, 'arable_land')
power_consumption_aus = clean_data(United_data, 'electric_power_comsumption')
forested_area_aus = clean_data(United_data, 'forest_area')
population_growth_aus = clean_data(United_data, 'population_growth')
urban_population_aus = clean_data(United_data, 'urban_population')
gdp_values_aus = clean_data(United_data, 'GDP')

# Creating a DataFrame from the cleaned data
minimum_length = min(len(electricity_access_aus), len(agriculture_land_aus), len(co2_emissions_aus),
                     len(arable_area_aus), len(power_consumption_aus), len(forested_area_aus),
                     len(population_growth_aus), len(urban_population_aus), len(gdp_values_aus))

clean_australia_data = pd.DataFrame({
    'Electricity Access': electricity_access_aus[:minimum_length],
    'Agricultural Land': agriculture_land_aus[:minimum_length],
    'CO2 Emissions': co2_emissions_aus[:minimum_length],
    'Arable Land': arable_area_aus[:minimum_length],
    'Power Consumption': power_consumption_aus[:minimum_length],
    'Forested Area': forested_area_aus[:minimum_length],
    'Population Growth': population_growth_aus[:minimum_length],
    'Urban Population': urban_population_aus[:minimum_length],
    'GDP': gdp_values_aus[:minimum_length]
})

# Display the first 5 rows of the clean data for Australia
print(clean_australia_data.head(5))

# Correlation heatmap for Australia data
corr_matrix_aus = clean_australia_data.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr_matrix_aus, annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap of United Kingdom")
plt.show()

  # Filter data for China
china_data = data[data['country'] == 'China']

# Function to clean data by removing null values
def clean_column_data(dataframe, column):
    """
   Cleans a specified column in a DataFrame by removing null values.

   Parameters:
   - dataframe: The DataFrame containing the data to be cleaned.
   - column: A string name of the column from which to remove null values.

   Returns:
   - A pandas Series with null values removed from the specified column.
   """
    return dataframe[column].dropna().values

# Cleaning data for various features
electricity_access_ch = clean_column_data(china_data, 'access_to_electricity')
agricultural_land_ch = clean_column_data(china_data, 'agricultural_land')
co2_emissions_ch = clean_column_data(china_data, 'co2_emission')
arable_land_ch = clean_column_data(china_data, 'arable_land')
power_consumption_ch = clean_column_data(china_data, 'electric_power_comsumption')
forest_area_ch = clean_column_data(china_data, 'forest_area')
population_growth_ch = clean_column_data(china_data, 'population_growth')
urban_population_ch = clean_column_data(china_data, 'urban_population')
gdp_ch = clean_column_data(china_data, 'GDP')

# Determine the minimum length for consistent DataFrame size
min_length = min(len(electricity_access_ch), len(agricultural_land_ch), len(co2_emissions_ch),
                 len(arable_land_ch), len(power_consumption_ch), len(forest_area_ch),
                 len(population_growth_ch), len(urban_population_ch), len(gdp_ch))

# Creating a DataFrame from the cleaned data
china_clean_data = pd.DataFrame({
    'Electricity Access': electricity_access_ch[:min_length],
    'Agricultural Land': agricultural_land_ch[:min_length],
    'CO2 Emissions': co2_emissions_ch[:min_length],
    'Arable Land': arable_land_ch[:min_length],
    'Power Consumption': power_consumption_ch[:min_length],
    'Forested Area': forest_area_ch[:min_length],
    'Population Growth': population_growth_ch[:min_length],
    'Urban Population': urban_population_ch[:min_length],
    'GDP': gdp_ch[:min_length]
})

# Display the first 5 rows of the clean data for China
print(china_clean_data.head(5))

# Correlation heatmap for China data
corr_matrix_china = china_clean_data.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr_matrix_china, annot=True, cmap="Blues")
plt.title("Correlation Heatmap of China")
plt.show()

corr_matrix_china 
# Filter data for India
india_data = data[data['country'] == 'India']

# Function to clean data by removing null values
def clean_india_data(column):
    """
  Cleans a specified column in the India data DataFrame by removing null values.

  This function specifically targets the india_data DataFrame, which is presumably 
  defined outside this function. It selects a column from the DataFrame, drops 
  all rows where the column has null values, and returns the cleaned column as a NumPy array. 
  This operation is useful for preparing the data for analysis that cannot handle NaN values.

  Parameters:
  - column: The name of the column in the india_data DataFrame to clean.

  Returns:
  - A NumPy array containing the non-null values of the specified column.
  """
    return india_data[column].dropna().values

# Cleaning data for different features
electricity_access_ind = clean_india_data('access_to_electricity')
agricultural_land_ind = clean_india_data('agricultural_land')
co2_emissions_ind = clean_india_data('co2_emission')
arable_land_ind = clean_india_data('arable_land')
power_consumption_ind = clean_india_data('electric_power_comsumption')
forest_area_ind = clean_india_data('forest_area')
population_growth_ind = clean_india_data('population_growth')
urban_population_ind = clean_india_data('urban_population')
gdp_ind = clean_india_data('GDP')

# Determine the minimum length for a consistent DataFrame size
min_length_ind = min(len(electricity_access_ind), len(agricultural_land_ind), len(co2_emissions_ind),
                     len(arable_land_ind), len(power_consumption_ind), len(forest_area_ind),
                     len(population_growth_ind), len(urban_population_ind), len(gdp_ind))

# Creating a DataFrame from the cleaned data
india_clean_data = pd.DataFrame({
    'Electricity Access': electricity_access_ind[:min_length_ind],
    'Agricultural Land': agricultural_land_ind[:min_length_ind],
    'CO2 Emissions': co2_emissions_ind[:min_length_ind],
    'Arable Land': arable_land_ind[:min_length_ind],
    'Power Consumption': power_consumption_ind[:min_length_ind],
    'Forested Area': forest_area_ind[:min_length_ind],
    'Population Growth': population_growth_ind[:min_length_ind],
    'Urban Population': urban_population_ind[:min_length_ind],
    'GDP': gdp_ind[:min_length_ind]
})

# Display the first 5 rows of the clean data for India
print(india_clean_data.head(5))

# Correlation heatmap for India data
corr_matrix_india = india_clean_data.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr_matrix_india, annot=True, cmap="Reds")
plt.title("Correlation Heatmap of India Data Features")
plt.show()

corr_matrix_india

data.columns

# Function to filter data by country and drop NaN values
def filter_data_by_country(dataframe, column, country):
    """
 Filters the DataFrame for a specific country and column, and drops any NaN values.

 Parameters:
 - dataframe: The DataFrame to filter.
 - column: The column to be included in the filtered DataFrame.
 - country: The country to filter the data by.

 Returns:
 - A filtered DataFrame with data for the specified country and column, with NaN values removed.
 """
    return dataframe[dataframe['country'] == country][['year', column]].dropna()

# Function to plot line charts
def plot_line_chart(data_dict, y_label, title):
    """
  Plots a line chart for the given data.

  Parameters:
  - data_dict: A dictionary with countries as keys and their respective DataFrames as values.
  - y_label: A label for the y-axis.
  - title: A title for the chart.

  The function creates a line chart for each country and adds it to the plot.
  """
    plt.figure(figsize=(15,10))
    for country, df in data_dict.items():
        plt.plot(df['year'], df[y_label], label=country)
    plt.xlabel('Year', fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.title(title)
    plt.legend()
    plt.show()

# Get CO2 emission data for each country
countries = ['Afghanistan', 'Bangladesh', 'China', 'India', 'Iran, Islamic Rep.', 'Kuwait', 
             'Pakistan', 'United Kingdom', 'Singapore', 'Ukraine', 'South Africa', 'New Zealand']
co2_emission_data = {country: filter_data_by_country(data, 'co2_emission', country) for country in countries}

# Plot CO2 Emission
plot_line_chart(co2_emission_data, 'co2_emission', 'CO2 Emission Over Years')

# Get Arable Land data for each country
arable_land_data = {country: filter_data_by_country(data, 'arable_land', country) for country in countries}

# Plot Arable Land
plot_line_chart(arable_land_data, 'arable_land', 'Arable Land Over Years')

# Function to filter electric power consumption data by year
def filter_data_by_year(dataframe, year):
    """
   Filters a DataFrame for entries corresponding to a specific year.

   Parameters:
   - dataframe: The DataFrame to filter.
   - year: The year to filter the data by.

   Returns:
   - A filtered DataFrame containing only data for the specified year.
   """
    return dataframe[dataframe['year'] == year]

# Filter electric power consumption for specific years
electric_power_data = data[['country', 'year', 'electric_power_comsumption']].dropna()
years = [1990, 2010, 2012, 2018]
electric_power_data_by_year = {year: filter_data_by_year(electric_power_data, year) for year in years}

# Display the data for each year
for year, df in electric_power_data_by_year.items():
    print(f"\nElectric Power Consumption Data for the year {year}:")
    print(df.head())