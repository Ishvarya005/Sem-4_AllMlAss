import numpy as np
import pandas as pd
import statistics
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path, sheet_name):
    # Load the data from a specific sheet in the Excel file
    return pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name)

def calculate_mean_and_variance(data):
    # Calculate the mean and variance of the Price data
    mean_price = statistics.mean(data.iloc[:, 4])
    variance_price = statistics.variance(data.iloc[:, 4])
    return mean_price, variance_price

def filter_and_calculate_mean(data, column_name, condition, label):
    # Filter data based on a condition and calculate the mean
    filtered_data = data[data[column_name] == condition]
    mean_price = statistics.mean(filtered_data['Price'])
    print(f"Mean of price data for {label}: {mean_price}")

def calculate_probability_of_loss(data):
    # Convert 'Chg%' column to string values
    data['Chg%'] = data['Chg%'].astype(str)

    # Extract Chg% values and convert to numerical format
    data['Chg%_num'] = data['Chg%'].str.rstrip('%').astype('float') / 100.0

    # Use lambda function to find negative values
    data['Loss'] = data['Chg%_num'].apply(lambda x: x < 0)

    # Calculate the probability of making a loss using a normal distribution assumption
    mean_chg = data['Chg%_num'].mean()
    std_chg = data['Chg%_num'].std()
    probability_of_loss = norm.cdf(0, loc=mean_chg, scale=std_chg)
    print(f"Probability of making a loss: {probability_of_loss:.2%}")

def calculate_probability_of_profit(data, day):
    # Filter rows for the specified day
    day_data = data[data['Day'] == day]

    # Use lambda function to find positive values
    day_data['Profit'] = day_data['Chg%_num'].apply(lambda x: x > 0)

    # Calculate the probability of making a profit using a normal distribution assumption
    mean_chg_day = day_data['Chg%_num'].mean()
    std_chg_day = day_data['Chg%_num'].std()
    probability_of_profit_day = 1 - norm.cdf(0, loc=mean_chg_day, scale=std_chg_day)
    print(f"Probability of making a profit on {day}: {probability_of_profit_day:.2%}")

def calculate_conditional_probability(data, day):
    # Filter rows for the specified day
    day_data = data[data['Day'] == day]

    # Use lambda function to find positive values
    day_data['Profit'] = day_data['Chg%_num'].apply(lambda x: x > 0)

    # Calculate the conditional probability of making a profit given that today is the specified day
    total_day_samples = day_data.shape[0]
    profit_given_day = day_data[day_data['Profit']].shape[0] / total_day_samples
    print(f"Conditional probability of making a profit given that today is {day}: {profit_given_day:.2%}")

def create_scatter_plot(data):
    # Create a scatter plot using seaborn
    sns.scatterplot(x='Day', y='Chg%_num', data=data, hue='Day', palette='viridis')

    # Set plot labels and title
    plt.xlabel('Day of the Week')
    plt.ylabel('Chg%')
    plt.title('Scatter Plot of Chg% Data Against Day of the Week')

    # Display the plot
    plt.show()

def main():
    file_path = "D:/Sem-4/ML/Assignments/Ass3/Lab Session1 Data.xlsx"
    sheet_name = 'IRCTC Stock Price'

    # Load data
    X = load_data(file_path, sheet_name)

    # Display loaded data
    print("Loaded Data:")
    print(X)

    # Calculate mean and variance of Price data
    mean_price, variance_price = calculate_mean_and_variance(X)
    print(f"Mean of Price data: {mean_price}")
    print(f"Variance of Price data: {variance_price}")

    # Filter and calculate mean for Wednesdays
    filter_and_calculate_mean(X, 'Day', 'Wed', 'all Wednesdays')

    # Filter and calculate mean for the month of April
    filter_and_calculate_mean(X, 'Month', 'Apr', 'the month of April')

    # Calculate probability of making a loss
    calculate_probability_of_loss(X)

    # Calculate probability of making a profit on Wednesday
    calculate_probability_of_profit(X, 'Wed')

    # Calculate conditional probability of making profit given that today is Wednesday
    calculate_conditional_probability(X, 'Wed')

    # Create scatter plot
    create_scatter_plot(X)

if __name__ == "__main__":
    main()
