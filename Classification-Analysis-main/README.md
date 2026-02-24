# Classification Analysis

## GUI Usage:

To get better understanding of dataset and result make sure to activate GUI.
```
git clone https://github.com/Cavid2002/Classification-Analysis
cd Classification-Analysis
pip install -r requirments.txt
streamlit run app.py
```


Once you run this command `streamlit run app.py` new browser window will be openned. Data set upload might take several seconds so be patient. Make sure to navigate the bars to observe the results. 


## Dataset Overview

The dataset contains 15,000 employee records designed for multi-class attrition risk prediction.

The target variable, Attrition_Risk_Level, is categorized into three classes: Low (0), Medium (1), and High (2) risk.

Features include demographic (Age, Gender, Education), compensation (Monthly_Income, Job_Role), engagement (Job_Satisfaction, Work_Life_Balance), workload (Num_Projects, Avg_Monthly_Hours), and career progression metrics (Years_at_Company, Promotions, Training).

The dataset is complete with no missing values, consists of 15 variables, and contains a mix of integer, float, and categorical data types.

Class distribution is moderately imbalanced: 50% Low Risk, 35% Medium Risk, and 15% High Risk.
