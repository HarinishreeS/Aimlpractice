from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Ensembling Methods
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Load Dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Base Models
dt = DecisionTreeClassifier()
svc = SVC(probability=True)
nb = GaussianNB()
lr = LogisticRegression(max_iter=1000)

# 1. Bagging
bagging = BaggingClassifier(base_estimator=dt, n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)
print("Bagging Accuracy:", accuracy_score(y_test, bagging.predict(X_test)))

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# 3. Boosting (AdaBoost)
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)
print("AdaBoost Accuracy:", accuracy_score(y_test, ada.predict(X_test)))

# 4. Voting (hard & soft)
voting_hard = VotingClassifier(estimators=[
    ('lr', lr), ('dt', dt), ('nb', nb)
], voting='hard')

voting_soft = VotingClassifier(estimators=[
    ('lr', lr), ('svc', svc), ('nb', nb)
], voting='soft')

voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)
print("Voting (Hard) Accuracy:", accuracy_score(y_test, voting_hard.predict(X_test)))
print("Voting (Soft) Accuracy:", accuracy_score(y_test, voting_soft.predict(X_test)))

# 5. Stacking
stack = StackingClassifier(
    estimators=[('lr', lr), ('svc', svc), ('nb', nb)],
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
print("Stacking Accuracy:", accuracy_score(y_test, stack.predict(X_test)))
