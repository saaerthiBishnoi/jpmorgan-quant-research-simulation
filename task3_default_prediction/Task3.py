import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

df = pd.read_csv('Task_3_and_4_Loan_Data.csv')

FEATURES = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding',
            'income', 'years_employed', 'fico_score']
TARGET = 'default'
RECOVERY_RATE = 0.10

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

test_probs = model.predict_proba(X_test_scaled)[:, 1]
test_preds = model.predict(X_test_scaled)
print("Model performance on held-out test set:")
print("  AUC:", round(roc_auc_score(y_test, test_probs), 4))
print("  Accuracy:", round(accuracy_score(y_test, test_preds), 4))

def predict_default_probability(credit_lines_outstanding, loan_amt_outstanding,
                                 total_debt_outstanding, income, years_employed,
                                 fico_score):
    borrower = pd.DataFrame([{
        'credit_lines_outstanding': credit_lines_outstanding,
        'loan_amt_outstanding': loan_amt_outstanding,
        'total_debt_outstanding': total_debt_outstanding,
        'income': income,
        'years_employed': years_employed,
        'fico_score': fico_score
    }])
    borrower_scaled = scaler.transform(borrower[FEATURES])
    prob_default = model.predict_proba(borrower_scaled)[0, 1]
    return prob_default

def expected_loss(credit_lines_outstanding, loan_amt_outstanding,
                   total_debt_outstanding, income, years_employed,
                   fico_score, recovery_rate=RECOVERY_RATE):
    pd_estimate = predict_default_probability(
        credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
        income, years_employed, fico_score
    )
    exposure_at_default = loan_amt_outstanding
    loss_given_default = 1 - recovery_rate
    loss = pd_estimate * exposure_at_default * loss_given_default
    return round(loss, 2)

if __name__ == "__main__":
    print("\nSample expected loss calculations:")

    sample_1 = dict(credit_lines_outstanding=0, loan_amt_outstanding=5000,
                     total_debt_outstanding=4000, income=78000,
                     years_employed=5, fico_score=650)
    print("Low-risk borrower:")
    print("  PD:", round(predict_default_probability(**sample_1), 4))
    print("  Expected Loss: $", expected_loss(**sample_1))

    sample_2 = dict(credit_lines_outstanding=5, loan_amt_outstanding=2000,
                     total_debt_outstanding=8000, income=26000,
                     years_employed=2, fico_score=570)
    print("High-risk borrower:")
    print("  PD:", round(predict_default_probability(**sample_2), 4))
    print("  Expected Loss: $", expected_loss(**sample_2))

    sample_3 = dict(credit_lines_outstanding=2, loan_amt_outstanding=10000,
                     total_debt_outstanding=5000, income=50000,
                     years_employed=3, fico_score=680)
    print("Medium-risk borrower:")
    print("  PD:", round(predict_default_probability(**sample_3), 4))
    print("  Expected Loss: $", expected_loss(**sample_3))
