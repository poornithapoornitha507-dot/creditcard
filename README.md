## Credit Card Default Risk Prediction
An end-to-end Data Science and Machine Learning project focused on analyzing consumer credit card transaction behavior and building a robust classification model to predict next-month payment defaults.
------------------------------
## 📌 Project Architecture
This project handles the complete product lifecycle of building a high-performing risk engine on consumer credit datasets. It addresses major challenges in real-world credit risk workflows, including severe class imbalance, extreme outlier variances, categorical feature encoding, and statistical feature selection.

[ Raw Credit Data ] ──> [ Data Cleaning & Mapping ] ──> [ Outlier Capping (IQR) ]
                                                                 │
[ Trained Classifier ] <── [ Standard Scaling ] <── [ Feature Selection ] <── [ Class Balancing (SMOTE) ]

------------------------------
## 📊 Phase 1: Exploratory Data Analysis (EDA)
The data pipeline utilizes a comprehensive suite of statistical analyses and diagnostic visualizations to clean and structure the data:
## 1. Structural Auditing & Data Profiling

* Dimensionality Check: Audited the initial matrix parameters to map rows, numerical predictors, and categorical indicators.
* Missing Value & Duplication Inspection: Verified complete data density to guarantee zero missing indices or transactional duplicates that would distort the modeling process.

## 2. Feature Re-Mapping & Contextualization

* Logical Renaming: Re-coded cryptic column headers representing multi-month payment timelines, statements, and payment values (e.g., PAY_0 to PAY_6, BILL_AMT1, PAY_AMT1) into intuitive chronological variables like SEP_PAY, AUG_BILL, and SEP_PAYMENT.
* Value Decoding: Translated integer class flags back into human-readable string matrices to improve downstream feature interpretation (e.g., mapping SEX to M/F, MARRIAGE to statuses like MARRIED/SINGLE, and the target values to a simple YES/NO).

## 3. Outlier Mitigation via IQR Capping

* High-spending card limits and rolling balances often create highly skewed distributions.
* Implemented an automated Interquartile Range (IQR) bounding function to compress anomalies:
* Calculated mathematical thresholds using Q1 - 1.5 * IQR as the lower limit and Q3 + 1.5 * IQR as the upper limit.
   * ExtExtreme values exceeding these bounds were capped to protect the model from variance instability without losing valuable record volume.

------------------------------
## ⚙️ Phase 2: Feature Engineering & Preprocessing
To ensure data compatibility for mathematical modeling, the dataset underwent structural transformations:

   1. One-Hot Encoding: Expanded high-cardinality categorical variables (EDUCATION and MARRIAGE) into distinct binary columns using Scikit-Learn’s OneHotEncoder(sparse_output=False) to prevent artificial ordering.
   2. Label Encoding: Converted binary variables (SEX and target) into normalized numeric indices using LabelEncoder.
   3. Handling Class Imbalance via SMOTE: Financial defaults are naturally rare, making the minority class heavily underrepresented. Implemented Synthetic Minority Over-sampling Technique (SMOTE) to synthesize minority instances and establish a 50/50 uniform class balance across the training distribution.
   4. ANOVA-based Feature Selection: Fitted a SelectKBest routine powered by an ANOVA F-value classifier (f_classif) to systematically score independent columns and retain the top 25 features holding the highest predictive weight.
   5. Standardization: Passed the optimized dataset through a StandardScaler to translate variable columns into zero mean and unit variance (μ = 0, σ = 1), neutralizing scale dominance during model fitting.

------------------------------
## 🤖 Phase 3: Machine Learning Modeling & Benchmarking
The preprocessed data was split into an 80:20 train-test configuration. A baseline Logistic Regression (LogisticRegression) model was selected for the risk evaluation layer, testing model performance against several classification metrics:
## Evaluation Metrics Breakdown

* Accuracy Score: Represents the overall ratio of correct predictions across both default and non-default states.
* Precision: Gauges the model's reliability in identifying actual high-risk accounts among all accounts flagged as default.
* Recall (Sensitivity): Determines the model's coverage in identifying the maximum number of true default incidents.
* F1-Score: The harmonic mean balancing precision and recall capabilities.

Classification Performance Output:
              precision    recall  f1-score   support

          NO       0.XX      0.XX      0.XX      XXXX
         YES       0.XX      0.XX      0.XX      XXXX

    accuracy                           0.XX     XXXXX
   macro avg       0.XX      0.XX      0.XX     XXXXX
weighted avg       0.XX      0.XX      0.XX     XXXXX

------------------------------
## 🏁 Strategic Project Takeaways## 1. Key Analytical Observations

* Behavior over Demographics: Multi-month bill tracking and delayed payment sequences carry far higher predictive weight for delinquency than demographic factors like age, gender, or education.
* Pipeline Dependability: Treating statistical outliers using the IQR bounding function combined with minority oversampling via SMOTE effectively prevented model bias toward the non-defaulting majority.

## 2. Financial & Operational Value

* Proactive Credit Risk Control: Deploying this model allows financial risk teams to isolate and flag high-risk accounts before the current billing cycle wraps up, enabling automated credit limit restrictions and minimizing bad debt.
* Production Viability: Since transaction records and payment delays are captured in real-time inside banking ledgers, this pipeline is ready to be directly embedded into live credit risk scoring engines.

------------------------------
## 🛠️ Technology Stack & Dependencies

* Language Platform: Python 3.x
* Core Analytics Framework: Pandas, NumPy
* Plotting & Visual Graphics Engine: Matplotlib, Seaborn
* Oversampling Engine: Imbalanced-Learn (SMOTE)
* Machine Learning Infrastructure: Scikit-Learn
* Preprocessing Tools: OneHotEncoder, LabelEncoder, StandardScaler, PowerTransformer
   * Selection Modules: SelectKBest (f_classif)
   * Classification Estimators: LogisticRegression


