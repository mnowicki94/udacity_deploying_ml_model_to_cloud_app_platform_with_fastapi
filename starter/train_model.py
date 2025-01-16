import joblib
from starter.ml.model import train_model, inference
from starter.ml.data import process_data

# Load the data
data = pd.read_csv("data/census.csv")

# Split the data
train, test = train_test_split(data, test_size=0.20)

# Process the data
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train the model
model = train_model(X_train, y_train)

# Save the model
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")
