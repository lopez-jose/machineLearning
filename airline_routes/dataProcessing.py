import pandas as pd
data = pd.read_excel('Data_Train.xlsx')  # Read in the data we downloaded
data.dropna(inplace=True)
import matplotlib.pyplot as plt
import seaborn as sns # Used for plotting
#import pandas as pd

### YOUR CODE HERE ###
#uncomment this after you use seaborn

#sns.countplot(x="Airline", data = data)
sns.countplot(data['Airline'])
plt.yticks(rotation = 20)
plt.xticks(rotation=90)

plt.show()
#Prints the first 5 values in the excel file
print(data.head())

airline_dict = {x:i for i, x in enumerate(data['Airline'].unique())}

print(airline_dict)               
def integer_encode_airline_dict(x):
  return airline_dict[x]                           
  

data['Airline']=data['Airline'].apply(integer_encode_airline_dict)

print(data.head())


source_dict = {y:i for i, y in enumerate(data['Source'].unique())}


def integer_encode_source_dict(y):
  return source_dict[y]

print(source_dict)

data['Source']=data['Source'].apply(integer_encode_source_dict)

destination_dict = {y:i for i, y in enumerate(data['Destination'].unique())}
print(destination_dict)
def integer_encode_destination_dict(y):
  return destination_dict[y]

data['Destination']=data['Destination'].apply(integer_encode_destination_dict)
print (destination_dict)

print(data.head())


# Step 1: Create a vocabulary for each column
# Step 2: Create a mapping function from each word in the vocabulary to a unique integer
# Step 3: Replce all words in the original data with the assigned integers



# Overwrite column to be in datatime format
# That is: YYYY/MM/DD -> YYYY-MM-DD
data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'])

data['day_of_week']=data['Date_of_Journey'].dt.day_name()
data['journey_month']=data['Date_of_Journey'].dt.month_name()

print(data.head())

day_dict = {x:i for i, x in enumerate(data['day_of_week'].unique())}

def integer_encode_day_dict(x):
    return day_dict[x]


data['day_of_week']=data['day_of_week'].apply(integer_encode_day_dict)


print(data.head())




month_dict = {x:i for i, x in enumerate(data['journey_month'].unique())}

def integer_encode_month_dict(x):
    return month_dict[x]

data['journey_month']=data['journey_month'].apply(integer_encode_month_dict)


print(data.head())


data['Route'].str.split(' ')

route_dict = {x: i for i, x in enumerate(data['Route'].unique())}


def integer_encode_route_dict(x):
    return route_dict[x]

data['Route']=data['Route'].apply(integer_encode_route_dict)


#We then implement one hot encoding on the routes that are taken. 
#data = pd.concat([data,pd.get_dummies(data['journey_month'],prefix = 'journey_month ')],axis = 1)
data = pd.concat([data,pd.get_dummies(data['Route'],prefix='Route ')],axis=1)

print(data.head())

def normalize_plot(col):
    return(col-col.min())/(col.max()-col.min())
data['Airline']=normalize_plot(data['Airline'])



data['Airline'].plot.hist(bins=50,title='Airline Plot')

plt.show()


