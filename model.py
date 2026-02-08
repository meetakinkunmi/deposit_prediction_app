# Importing packages
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load data
df = pd.read_excel("bank-full.xlsx")
print(df.head())

# Separating classes for upsampling
not_deposit = df[df["y"] == 'no']
deposit = df[df["y"] == 'yes']

# Upsample deposit
deposit_upsampled = pd.DataFrame(resample(deposit,
                          replace=True, # sample with replacement (we need to duplicate observations)
                          n_samples=len(not_deposit), # match number in minority class
                          random_state=30)) # reproducible results

# Combine upsampled minority class with majority class
upsampled = pd.concat([deposit_upsampled, not_deposit], axis=0)

# Splitting the data into target and predictors
y = upsampled['y']

X = upsampled.drop("y", axis=1)

# Splitting the data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)

# Separating numerical and categorical features
X_num = X.select_dtypes(include=['number']).columns
X_cat = X.select_dtypes(include=['str']).columns

# Scaling numerical and encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), X_num),
        ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), X_cat)
    ]
)

# Pipeline to handle preprocess and classification
pipe = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('clasiffier', LogisticRegression(max_iter=500))
    ]
)

# Fit the classification model
pipe.fit(X_train, y_train)

# Make pickle file of the model
pickle.dump(pipe, open("model.pkl", "wb"))