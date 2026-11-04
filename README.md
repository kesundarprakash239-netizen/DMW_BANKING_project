# DMW_BANKING_project
Activity based on banking system (dmw)
# Bank Marketing Response and Customer Segmentation

Exploratory data analysis, campaign-response prediction,
association-rule mining, customer segmentation, and OLAP reporting using
the **UCI Bank Marketing** dataset.

## Problem Statement

Build a campaign-analysis system to identify patterns associated with
customer subscription to a term deposit and to discover customer groups
that respond differently to marketing activity.

The project examines demographic, contact, campaign, economic and
outcome attributes. It evaluates data quality and preprocessing choices,
compares dimensionality-reduction and feature-selection methods,
develops classification models, discovers frequent patterns, performs
customer clustering, and designs a campaign data warehouse with OLAP
analysis.

## Dataset

-   **Source:** [UCI Machine Learning Repository -- Bank
    Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
-   **Dataset file:** `bank-additional-full.csv`
-   **Records:** 41,188
-   **Attributes:** 21, including the target attribute `y`
-   **Target:** `y` --- whether the customer subscribed to a term
    deposit (`yes` or `no`)
-   **Format:** CSV with semicolon (`;`) separators

### Attribute groups

-   **Demographic:** `age`, `job`, `marital`, `education`
-   **Customer financial status:** `default`, `housing`, `loan`
-   **Contact details:** `contact`, `month`, `day_of_week`
-   **Campaign history:** `campaign`, `pdays`, `previous`, `poutcome`
-   **Economic indicators:** `emp.var.rate`, `cons.price.idx`,
    `cons.conf.idx`, `euribor3m`, `nr.employed`
-   **Call information:** `duration`
-   **Outcome:** `y`

> **Important:** `duration` is known only after a call has taken place.
> It is excluded from the primary pre-call prediction model to avoid
> information leakage.

## What the Project Does

1.  **Loads and inspects the dataset**
    -   Reads the semicolon-separated CSV file.
    -   Checks dimensions, data types, target distribution and data
        quality.
    -   Identifies explicit `unknown` category values and conventional
        missing values.
2.  **Exploratory data analysis**
    -   Summarizes demographic, contact, campaign and economic
        attributes.
    -   Visualizes distributions, boxplots and numerical correlations.
    -   Compares subscription rates across customer groups, contact
        channels and campaign periods.
    -   Examines repeated contacts and their association with
        subscription outcomes.
3.  **Data preprocessing**
    -   Encodes categorical attributes and scales numerical attributes
        where required.
    -   Inspects outliers and special values, including `pdays = 999`
        (not previously contacted).
    -   Examines redundancy and correlated attributes.
    -   Splits data before fitting preprocessing transformations.
4.  **PCA and attribute-subset selection**
    -   Applies Principal Component Analysis (PCA).
    -   Applies attribute-subset selection.
    -   Compares retained features/components and classification
        performance.
    -   Fits transformations on training data only to avoid leakage.
5.  **Classification**
    -   **Decision Tree:** learns decision paths from customer and
        campaign attributes.
    -   **Rule-based classifier:** learns interpretable IF--THEN rules
        (e.g. RIPPER/JRip).
    -   **K-Nearest Neighbours (KNN):** predicts outcomes using similar
        customer records.
    -   Compares accuracy, precision, recall, F1-score and confusion
        matrices.
6.  **Class imbalance, thresholds and repeated contacts**
    -   Measures the imbalance between `yes` and `no` outcomes.
    -   Evaluates training-only resampling or class weighting where
        appropriate.
    -   Studies how changing the classification threshold affects
        precision and recall.
    -   Examines how subscription rates vary with the number of campaign
        contacts.
7.  **Association rule mining**
    -   Converts selected customer and campaign attributes into
        transaction-style item sets.
    -   Uses Apriori and FP-Growth to find frequent itemsets.
    -   Evaluates rules using support, confidence and lift.
    -   Interprets rules as associations, not causal effects.
8.  **Customer segmentation**
    -   Applies K-Means, hierarchical clustering and DBSCAN.
    -   Justifies representation and distance handling for mixed
        numerical and categorical attributes.
    -   Profiles clusters using customer, campaign and subscription
        summaries.
9.  **Data warehouse and OLAP**
    -   Designs a star schema with customer, campaign, contact-period
        and outcome dimensions.
    -   Defines campaign-contact facts and subscription measures.
    -   Uses OLAP-style aggregation to compare subscription rates across
        segments and campaign periods.
10. **Responsible use**
    -   Considers customer privacy and data minimization.
    -   Discusses potential discriminatory targeting and fairness.
    -   Distinguishes prediction of subscription from the causal impact
        of marketing contact.

## Expected Outcomes

-   A comparative evaluation of campaign-response classifiers.
-   An analysis of data quality, preprocessing, PCA and attribute
    selection.
-   Interpretable association rules describing customer and campaign
    patterns.
-   Customer segments with documented profiles and clustering
    limitations.
-   A campaign warehouse design and OLAP reporting layer.
-   A discussion of privacy, fairness and the limitations of predictive
    conclusions.

## Results

Populate final metric tables from the executed experiments. Do not
present expected or illustrative values as measured results.

  -----------------------------------------------------------------------
  Analysis                            Measures to report
  ----------------------------------- -----------------------------------
  Classification                      Accuracy, precision, recall,
                                      F1-score, confusion matrix

  Threshold analysis                  Threshold, precision, recall and
                                      F1-score

  PCA vs. feature selection           Number of retained
                                      features/components and classifier
                                      metrics

  Association mining                  Support, confidence and lift

  Clustering                          Cluster sizes, silhouette score
                                      where applicable, and cluster
                                      profiles

  OLAP                                Subscription counts and
                                      subscription rates by segment and
                                      campaign period
  -----------------------------------------------------------------------

## Suggested Repository Structure

``` text
├── README.md
├── bank-additional-full.csv
├── bank_marketing.py
├── notebooks/
│   └── bank_marketing_analysis.ipynb
├── plots/
│   ├── target_distribution.png
│   ├── numerical_distributions.png
│   ├── boxplots.png
│   └── correlation_heatmap.png
├── results/
│   ├── classification_results.csv
│   ├── association_rules.csv
│   └── cluster_profiles.csv
└── warehouse/
    ├── schema.sql
    └── olap_queries.sql
```

The structure is a suggestion. Keep only files that actually exist in
your repository.

## Getting Started

### 1. Clone the repository

``` bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. (Optional) Create a virtual environment

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\\Scripts\\activate
```

macOS/Linux:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn mlxtend scipy
```

Install and configure Weka separately if using its GUI-based tools.

### 4. Run the analysis

If your main script is named `bank_marketing.py`:

``` bash
python bank_marketing.py
```

If using Google Colab, upload `bank-additional-full.csv` and execute the
notebook cells in order.

## Evaluation and Methodological Notes

-   Use a stratified train/test split for classification.
-   Fit encoders, scalers, PCA and feature selection on training data
    only. For cross-validation, put these steps inside each fold using a
    pipeline.
-   Apply SMOTE only to training data, never to the held-out test set.
-   Because the target is imbalanced, report positive-class precision,
    recall and F1-score instead of relying on accuracy alone.
-   Treat `pdays = 999` as a special code meaning the customer was not
    previously contacted, not as an ordinary number of days.
-   Repeated-contact patterns are observational; they do not prove that
    additional calls cause a change in subscription probability.
-   Prediction estimates patterns in historical data. Causal campaign
    impact requires an appropriate experimental or causal-inference
    design.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn,
imbalanced-learn, mlxtend, SciPy, Weka, Google Colab and Jupyter
Notebook.

## Citation

Moro, S., Cortez, P., and Rita, P. (2014). *A Data-Driven Approach to
Predict the Success of Bank Telemarketing*. Decision Support Systems,
62, 22--31.

UCI Machine Learning Repository. *Bank Marketing*.
https://archive.ics.uci.edu/dataset/222/bank+marketing
