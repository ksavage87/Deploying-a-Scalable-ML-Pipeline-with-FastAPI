# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

##Github Link
https://github.com/ksavage87/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git


## Model Details

The model type used was the Random Forest Classifier. Random Forest Classifier is an ensemble learning method that constructs numerous decision trees, then collects the predictions of the trees and selects the best model of the individual trees. All of the hyperparameters were set to default.

## Intended Use

This model is intended to predict salary ranges based upon demographic census data. It provides understanding of factors contributing to salary ranges and can be used for application such as social sciences research, workforce planning, or human resource analytics.

## Training Data

The training data was gathered from the census.csv file provided. The data in the file contained US census information about incomes.The training data consists of demographic data about employment, incomde, age, education level, occupation, etc. Prior to training, the data was preprocessed to handle missing values and encode categorical variables.

## Evaluation Data

The census income data used for training was also used for the evulation data.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Three metrics were used; precision, recall, and F1 score. Precision scored 0.7394. Recall scored 0.6410. F1 scored 0.6867.

## Ethical Considerations

In order to be ethical we should consider fairness, privacy, and trnasparency. The model should be exaluated for fairness in race, gender, and age to rule out biases. It should also hide sensitive demographic data and any PII from the model. The model should offer transparency about its inputs, outputs, and decision making process like the Random Forest Classifier.

## Caveats and Recommendations

The model's performance will depend on the quality of the training data. The model should be monitored and updated each time the census is taken. 