<h2 style="color:red;">✅ Tuning the decision threshold for class prediction</h2>

Classification is best divided into two parts:

- the statistical problem of learning a model to predict, ideally, class probabilities;

- the decision problem to take concrete action based on those probability predictions.

Let’s take a straightforward example related to weather forecasting: the first point is related to answering “what is the chance that it will rain tomorrow?” while the second point is related to answering “should I take an umbrella tomorrow?”.


When it comes to the scikit-learn API, the first point is addressed by providing scores using ```predict_proba``` or ```decision_function```. 

The former returns conditional probability estimates P(y/X) for each class, while the latter returns a decision score for each class.

- The decision corresponding to the labels is obtained with ```predict```.

- In binary classification, a decision rule or action is then defined by thresholding the scores, leading to the prediction of a single class label for each sample.

For binary classification in scikit-learn, class labels predictions are obtained by hard-coded cut-off rules: a positive class is predicted when the conditional probability P(y/X) is greater than 0.5 (obtained with predict_proba) or if the decision score is greater than 0 (obtained with decision_function).


Here, we show an example that illustrates the relatonship between conditional probability estimatesP(y/X) and class labels:

```
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
X, y = make_classification(random_state=0)
classifier = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
classifier.predict_proba(X[:4])
classifier.predict(X[:4])
```

While these hard-coded rules might at first seem reasonable as default behavior, they are most certainly not ideal for most use cases. Let’s illustrate with an example.

Consider a scenario where a predictive model is being deployed to assist physicians in detecting tumors. In this setting, physicians will most likely be interested in identifying all patients with cancer and not missing anyone with cancer so that they can provide them with the right treatment. In other words, physicians prioritize achieving a high recall rate. This emphasis on recall comes, of course, with the trade-off of potentially more false-positive predictions, reducing the precision of the model. That is a risk physicians are willing to take because the cost of a missed cancer is much higher than the cost of further diagnostic tests. Consequently, when it comes to deciding whether to classify a patient as having cancer or not, it may be more beneficial to classify them as positive for cancer when the conditional probability estimate is much lower than 0.5.


## Post-tuning the decision threshold

