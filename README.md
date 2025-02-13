# 🚀 Titanic Survival Prediction with Random Forest

This project predicts the survival of Titanic passengers using a **Random Forest Classifier**. It includes data preprocessing, model training, evaluation, and visualization of results.

---

## 📂 Project Structure
```
├── data/
│   ├── train.csv
│   ├── test.csv
├── model/
│   ├── trained_model.pkl
├── scripts/
│   ├── train_model.py
│   ├── predict.py
├── results/
│   ├── predictions.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curve.png
├── README.md
```

---

## 📊 Dataset Information
- **Source:** Kaggle Titanic Dataset
- **Features Used:**
  - Pclass (Passenger Class)
  - Sex (Gender)
  - Age
  - SibSp (Number of Siblings/Spouses)
  - Parch (Number of Parents/Children)
  - Fare (Ticket Fare)
  - Embarked (Port of Embarkation)
- **Target:** `Survived` (0 = No, 1 = Yes)

---

## 🔧 Data Preprocessing
- **Dropped Unnecessary Columns:** `Ticket`
- **Handled Missing Values:**
  - Numeric Columns → Filled with Mean
  - Categorical Columns → Filled with Mode
- **Encoded Categorical Variables:** `Sex`, `Embarked`

```python
# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)
```

---

## 🏆 Model Training (Random Forest Classifier)
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

---

## 🎯 Model Evaluation
### ✅ **Confusion Matrix**
![Confusion Matrix](results/confusion_matrix.png)
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.show()
```

### ✅ **Feature Importance**
![Feature Importance](results/feature_importance.png)
```python
importances = rf_model.feature_importances_
sns.barplot(x=importances, y=X_train.columns)
plt.show()
```

### ✅ **ROC Curve**
![ROC Curve](results/roc_curve.png)
```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.legend()
plt.show()
```

---

## 📌 Using the Model for Prediction
### **Predict on New Data & Save Results**
```python
new_df = pd.read_csv("new_data.csv")
predictions = rf_model.predict(new_df)
new_df["Predicted_Survived"] = predictions
new_df.to_csv("predictions.csv", index=False)
```

---

## 📤 Deployment
1. **Clone the Repo:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Training:**
   ```bash
   python scripts/train_model.py
   ```
4. **Run Prediction:**
   ```bash
   python scripts/predict.py
   ```

---

## 📜 License
This project is licensed under the MIT License.

