""" Decision function
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris

cancer = load_iris()
X, y = cancer.data, cancer.target

train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, random_state=33)

#max_features = sqrt(train_y)
gradient = GradientBoostingClassifier(random_state=42, max_depth=1, learning_rate=0.01)
gradient.fit(train_x, train_y)

print (gradient.decision_function(X_test))
print (gradient.predict_proba(X_test))