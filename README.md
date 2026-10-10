
# 2. `README.md` — PROJECT GUARDRAIL

I recommend putting this in the root of your repository as `README.md`.

````markdown
# Machine Learning-Based Bank Customer Relationship Exit Prediction

## University Project

A machine-learning-based decision support system designed to identify banking customers who are at increased risk of exiting their relationship with a bank.

---

# 1. Project Overview

Customer relationship exit is an important problem in the banking sector because losing an existing customer can affect deposits, transaction activity, product relationships, and long-term customer value.

This project aims to develop a machine-learning-based system that uses customer relationship, financial, transaction, engagement, and service-friction information to estimate the probability that a customer will exit their banking relationship within a future prediction window.

The project will initially focus on:

> **Binary relationship-exit classification**

and will progressively evolve into:

```text
Relationship Exit Prediction
          ↓
Exit Probability
          ↓
Risk Classification
          ↓
Customer Prioritization
          ↓
Decision Support
          ↓
Prescriptive Recommendations
````

---

# 2. Academic Scope

This project is intended as a university academic project.

It is NOT:

* a production banking system
* a real banking customer-monitoring system
* a financial advisory system
* a regulatory compliance system
* a replacement for human decision-making

The purpose is to demonstrate:

* data engineering
* machine learning
* feature engineering
* predictive modeling
* model evaluation
* explainable AI
* decision-support design

---

# 3. Primary Objective

Predict whether a customer is likely to exit their banking relationship within the selected prediction horizon.

Initial prediction horizon:

```text
6 months
```

The architecture must support:

```text
6 months
↓
12 months
```

without requiring a complete redesign.

---

# 4. Target Variable

## `relationship_exit`

Binary target:

| Value | Meaning                                                       |
| ----: | ------------------------------------------------------------- |
|     0 | Customer did not exit during the prediction window            |
|     1 | Customer exited the relationship during the prediction window |

The initial implementation uses a 6-month future outcome window.

---

# 5. Relationship Exit Definition

For this project, relationship exit refers to the discontinuation of the customer's meaningful relationship with the bank.

Because the project uses synthetic data, the exact operational definition of exit is explicitly defined during dataset generation.

The definition must remain consistent across:

* dataset generation
* model training
* evaluation
* documentation

---

# 6. Dataset Policy

## 6.1 Public Dataset Search

Before generating synthetic data, the project should investigate whether a suitable public dataset exists.

A public dataset may be used if it:

1. has an academically defensible source
2. contains a sufficiently large number of relevant variables
3. has a meaningful churn/exit target
4. does not introduce unacceptable leakage
5. can be legally and ethically used for academic purposes

An external dataset should NOT be used simply because it is popular.

---

# 7. Synthetic Dataset Policy

If no sufficiently suitable public dataset exists, synthetic data will be used.

The synthetic dataset must be explicitly identified as:

> **Synthetic / simulated banking data**

It must NEVER be represented as actual customer data.

The synthetic data should use:

* domain-informed distributions
* realistic feature relationships
* temporal behavior
* noise
* outliers
* missing values
* customer heterogeneity
* class overlap

The objective is to create a realistic ML experimentation environment, not to imitate any specific bank's customers.

---

# 8. Initial Dataset Size

Development dataset:

```text
10,000 customers
```

This dataset is used for:

* pipeline development
* debugging
* feature validation
* baseline modeling
* initial experimentation

After validation, the dataset may be scaled to:

```text
50,000+
100,000+
```

The larger dataset should only be generated after the methodology works correctly on the smaller dataset.

---

# 9. Prediction Architecture

The project follows a temporal prediction structure.

```text
                 OBSERVATION WINDOW
              Historical information
                       │
                       ▼
              Feature Engineering
                       │
                       ▼
                 PREDICTION DATE
                       │
                       ▼
              Model Prediction
                       │
                       ▼
             FUTURE 6-MONTH WINDOW
                       │
                       ▼
              Relationship Exit
```

The model may only use information available before or at the prediction date.

---

# 10. DATA LEAKAGE POLICY

## HARD RULE

No future information may be used as a predictor.

### Allowed

```text
Previous balance trend
Previous transaction decline
Previous complaints
Previous service interactions
Previous digital activity
```

### Not allowed

```text
Future account closure
Future transaction activity
Future balance
Future complaints
Future service interactions
```

The target is the only variable allowed to depend on the future outcome window.

Every feature should have a clearly defined temporal meaning.

---

# 11. Required Feature Set

The project contains 30 core parameters.

## A. Customer & Relationship Context

1. Age Band
2. Customer Tenure
3. Customer Segment
4. Number of Active Products
5. Primary Account Type
6. Relationship Status

---

## B. Financial Position

7. Average Account Balance
8. Balance Trend
9. Income/Salary Inflow Regularity
10. Loan Repayment Status
11. Credit Utilization
12. Recent Overdraft Occurrences

---

## C. Transaction Behavior

13. Transaction Count
14. Average Transaction Value
15. Transaction Value Trend
16. Inflow-to-Outflow Ratio
17. Recurring Payment Activity
18. Failed/Declined Transaction Rate

---

## D. Engagement & Channel Usage

19. Days Since Last Meaningful Interaction
20. Digital Banking Activity Level
21. Change in Digital Activity
22. Number of Active Banking Channels
23. Branch/Service Interaction Frequency
24. Unusual Reduction in Account Usage

---

## E. Service & Relationship Friction

25. Complaint Occurrence
26. Complaint Resolution Time
27. First-Contact Resolution
28. Service Interaction Frequency
29. Failed Service / Transaction Incidents
30. Service Channel Wait Time

---

# 12. Temporal Data Structure

Where possible, raw data should contain historical monthly information.

Conceptually:

```text
Customer
   │
   ├── Profile
   │
   ├── Financial history
   │
   ├── Transaction history
   │
   ├── Engagement history
   │
   └── Service history
            │
            ▼
     Observation Window
            │
            ▼
       Prediction Date
            │
            ▼
       Outcome Window
```

This allows features such as:

* balance trend
* transaction trend
* digital activity change
* unusual activity reduction

to be calculated rather than randomly assigned.

---

# 13. Feature Engineering Principles

Derived features may include:

* balance change percentage
* transaction decline percentage
* transaction frequency change
* inflow/outflow ratio
* product penetration
* channel diversity
* service friction score
* complaint recurrence
* failed transaction frequency
* digital engagement decline
* activity recency
* behavioral volatility
* transaction consistency
* balance volatility
* engagement trend

Every engineered feature must have:

1. a clear definition
2. a formula
3. a reason for inclusion
4. a leakage check
5. an evaluation

---

# 14. Synthetic Target Generation

The target must NOT be generated using a simple deterministic rule such as:

```python
if complaints > X:
    churn = 1
```

Instead, synthetic customer behavior should contribute to an underlying latent relationship-health or exit-propensity process.

Conceptually:

```text
Customer Characteristics
          +
Financial Behavior
          +
Transaction Behavior
          +
Engagement
          +
Service Friction
          ↓
Latent Relationship Health
          ↓
Exit Propensity
          ↓
Probabilistic Relationship Exit
```

The latent variable itself must NOT be included as a model feature.

Noise and exceptions must be present.

---

# 15. Machine Learning Strategy

## Phase 1 — Baseline

Start with:

* Dummy Classifier
* Logistic Regression

## Phase 2 — Tree-Based Models

Evaluate:

* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost, where appropriate

## Phase 3 — Optimization

Perform hyperparameter tuning only after identifying promising models.

## Phase 4 — Explainability

Use:

* feature importance
* permutation importance
* SHAP where appropriate

---

# 16. Evaluation Metrics

Accuracy is NOT sufficient.

Primary metrics:

* Recall
* Precision
* F1 Score
* PR-AUC

Secondary metrics:

* ROC-AUC
* Accuracy
* Specificity
* Confusion Matrix
* Calibration

The selected model must be justified based on the project's objective rather than simply selecting the model with the highest accuracy.

---

# 17. Class Imbalance

Customer exit may represent a minority class.

Therefore, investigate:

* class weighting
* threshold tuning
* SMOTE
* other appropriate techniques

SMOTE must only be applied to training data.

No synthetic oversampling may be applied to validation or test data.

---

# 18. Train / Validation / Test Policy

The test dataset must remain untouched until final evaluation.

Do NOT use test data for:

* feature selection
* hyperparameter tuning
* threshold selection
* model selection

Where appropriate, temporal validation should be preferred.

---

# 19. Model Interpretability

The final model should provide more than:

```text
Customer A → 82% risk
```

It should ideally provide:

```text
Customer A
Exit Probability: 82%

Major contributing factors:
- declining transaction activity
- reduced digital engagement
- negative balance trend
- increased service interactions
- repeated failed transactions
```

Interpretability is an important component of the decision-support system.

---

# 20. Decision-Support Roadmap

The system will be developed incrementally.

### Stage 1

```text
Predict Exit
```

### Stage 2

```text
Predict Exit Probability
```

### Stage 3

```text
Assign Risk Category
```

Example:

```text
Low
Moderate
High
Critical
```

Thresholds must be validated rather than assumed.

### Stage 4

```text
Prioritize Customers
```

### Stage 5

```text
Recommend Intervention
```

### Stage 6

```text
Prescriptive Optimization
```

No later stage should be developed before the previous stage has been validated.

---

# 21. Recommended Decision-Support Architecture

```text
                CUSTOMER DATA
                     │
                     ▼
              ETL PIPELINE
                     │
                     ▼
          FEATURE ENGINEERING
                     │
                     ▼
            CHURN CLASSIFIER
                     │
                     ▼
           EXIT PROBABILITY
                     │
                     ▼
             RISK SCORING
                     │
                     ▼
        CUSTOMER PRIORITIZATION
                     │
                     ▼
       KEY RISK FACTOR EXPLANATION
                     │
                     ▼
        RECOMMENDED INTERVENTION
```

---

# 22. Project Folder Structure

```text
bank-churn-decision-support/
│
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_data_quality_and_eda.ipynb
│   ├── 03_etl_and_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_model_tuning.ipynb
│   ├── 07_model_interpretability.ipynb
│   └── 08_decision_support.ipynb
│
├── src/
│   ├── data_generation/
│   ├── etl/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── decision_support/
│
├── models/
├── reports/
│   ├── figures/
│   └── results/
│
└── docs/
```

---

# 23. Reproducibility

All experiments must use controlled random seeds.

Default:

```python
RANDOM_STATE = 42
```

Dataset generation must be reproducible.

The same configuration and seed should produce the same dataset.

---

# 24. Experiment Tracking

Every model experiment should record:

| Parameter       | Required    |
| --------------- | ----------- |
| Model           | Yes         |
| Features        | Yes         |
| Hyperparameters | Yes         |
| Class balancing | Yes         |
| Threshold       | Yes         |
| Precision       | Yes         |
| Recall          | Yes         |
| F1              | Yes         |
| ROC-AUC         | Yes         |
| PR-AUC          | Yes         |
| Accuracy        | Yes         |
| Training time   | Recommended |

Results should be saved rather than existing only inside notebooks.

---

# 25. Data Quality Checks

Before modeling, check:

* missing values
* duplicate customers
* invalid categorical values
* impossible numerical values
* invalid ratios
* extreme outliers
* temporal inconsistencies
* target imbalance
* feature leakage
* duplicate observations

---

# 26. Academic Integrity Rules

The project must never:

* claim synthetic data is real
* invent model results
* invent sources
* fabricate citations
* report untested claims as facts
* hide data-generation assumptions
* intentionally introduce leakage
* select a model solely because it produces impressive metrics

If an experiment has not been performed, label it:

> Not yet evaluated.

---

# 27. Scalability Strategy

The initial dataset contains:

```text
10,000 customers
```

Once the complete pipeline is validated:

```text
10K
 ↓
50K
 ↓
100K+
```

Scaling should require minimal code changes.

Dataset size must therefore be configuration-driven.

---

# 28. Prediction Horizon Strategy

Initial:

```text
6 months
```

Future experiment:

```text
12 months
```

The horizon should be controlled through configuration.

Example:

```python
PREDICTION_HORIZON_MONTHS = 6
```

Changing it to:

```python
PREDICTION_HORIZON_MONTHS = 12
```

should not require rewriting the entire pipeline.

---

# 29. Definition of Done

The project is considered complete only when it contains:

[ ] Reproducible dataset generation or documented external dataset

[ ] Data-quality validation

[ ] ETL pipeline

[ ] 30 required features

[ ] Clearly defined target

[ ] Leakage checks

[ ] Baseline model

[ ] Multiple model comparison

[ ] Hyperparameter tuning

[ ] Proper evaluation

[ ] Explainability

[ ] Exit probability

[ ] Risk categorization

[ ] Customer prioritization

[ ] Decision-support recommendations

[ ] Final documentation

[ ] Reproducible results

---

# 30. Development Philosophy

The project follows:

```text
CORRECTNESS
    >
DATA QUALITY
    >
METHODOLOGY
    >
INTERPRETABILITY
    >
REPRODUCIBILITY
    >
PERFORMANCE
    >
COMPLEXITY
```

Complexity should only be added when it provides measurable value.

---

# 31. Current Development Stage

## CURRENT STAGE: PHASE 0 → PHASE 1

Current objectives:

1. Establish project structure
2. Investigate available datasets
3. Determine whether synthetic data is required
4. Create the 10,000-customer development dataset
5. Validate the generated data
6. Document assumptions

DO NOT start advanced machine learning until the dataset and data-generation methodology have been validated.

---

# 32. Next Milestone

The immediate milestone is:

> **Generate and validate the first 10,000-customer dataset with the required 30 parameters and relationship-exit target.**

Only after this milestone is validated should model development begin.

````

---

## 3. One architectural choice I strongly recommend

There is one thing I'd change from a typical "churn dataset" project.

**Don't make your final dataset just this:**

```text
Customer_ID | Age | Balance | Transactions | Complaints | ... | Churn
````

Instead, think of it as **three layers**:

```text
                 ┌──────────────────────┐
                 │     RAW SIMULATION   │
                 │                      │
                 │ Customer profile     │
                 │ Monthly financials   │
                 │ Transactions         │
                 │ Engagement           │
                 │ Service interactions │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     FEATURE STORE    │
                 │                      │
                 │ 30 required features │
                 │ + engineered features│
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     ML DATASET       │
                 │                      │
                 │ X = past information │
                 │ y = future exit      │
                 └──────────────────────┘
```

This is **much stronger academically**.

For example, your `balance_trend` shouldn't be an arbitrary number generated independently. It should emerge from something like:

```text
Month 1    ₹82,000
Month 2    ₹79,500
Month 3    ₹76,200
Month 4    ₹71,800
Month 5    ₹69,100
Month 6    ₹63,400
                  ↓
       balance_trend = -22.7%
```

Likewise:

```text
Digital activity

Month 1 → 31 sessions
Month 2 → 29
Month 3 → 27
Month 4 → 22
Month 5 → 17
Month 6 → 11

             ↓

change_in_digital_activity = -64.5%
```

That gives your project a **real ETL + feature engineering story**, rather than "we generated 30 columns and trained Random Forest."

---

## 4. One more important thing: don't chase 99% accuracy

Because **we control the synthetic data**, we could accidentally make the problem artificially easy.

For example, if we generate:

```text
High complaints → 90% churn
Low complaints  → 5% churn
```

the model might achieve 97–99% accuracy.

That sounds impressive but actually makes the project **worse**, because a professor can reasonably ask:

> "Did you create the data so that the model already knew the answer?"

Instead, we want something more like:

```text
                  Relationship Exit
                         ▲
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
 Transaction        Engagement        Service
 Behavior           Decline           Friction
       │                 │                 │
       └────────────┬────┴────┬────────────┘
                    │         │
                 Financial  Customer
                  Position   Context
                    │         │
                    └────┬────┘
                         │
                       Noise
                         │
                         ▼
                  EXIT PROPENSITY
```

So some customers with **bad behavior won't leave**, and some customers with **apparently healthy behavior will leave**.

That's what gives us a meaningful ML problem.

---

### Our development path is now locked
## Known limitations

- **`branch_service_interaction_frequency` is constant (0) in the dataset.**
  Verified: all 120,000 rows in `data/raw/monthly_engagement.csv` are 0, and
  `ml_ready_dataset.csv` has 1 unique value for this column. The model therefore
  cannot learn from it, so it effectively uses 29 of 30 features.
  Suspected cause: the formula in `generate_synthetic_dataset.py` (line 156) goes
  far below 0 and is clipped to 0. Not yet confirmed or fixed.
- `service_interaction_frequency` is derived from this feature (line 162), so it
  is probably affected too.
- The data quality report did not flag zero-variance columns.

