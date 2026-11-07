# Predictive Modeling and Optimization in Logistics Systems

## Project Overview
This project focuses on predicting shipment transit times using Machine Learning techniques. It helps logistics companies understand delivery delays and improve transportation planning.

## Objective
To develop a predictive model that estimates shipment transit time based on distance, traffic congestion, carrier type, weather conditions, and dispatch shift.

## Dataset
A simulated dataset containing 10,000 shipment records.

Features:
- Distance_Miles
- Congestion_Index
- Carrier_Tier
- Weather_Condition
- Dispatch_Shift

Target:
- Final_Transit_Time

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Machine Learning Algorithms
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## Model Evaluation Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 1.605 | 2.021 | 0.957 |
| Decision Tree | 2.387 | 3.007 | 0.904 |
| Random Forest | 1.820 | 2.264 | 0.945 |

Linear Regression achieved the best test performance among the three models.

## Project Structure

```text
logistics-predictive-modeling/
├── data/
├── generated/
├── models/
├── predictive_modeling.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

1. Install Python.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the Python program:

```bash
python predictive_modeling.py
```

## Optimization Strategies
- Dynamic carrier allocation.
- Monitoring traffic and weather conditions.
- Proactive workforce planning.
- Reducing shipment delays.

## Conclusion
This project demonstrates how Machine Learning can help predict logistics transit times and support better transportation decisions.

**Project:** YuvaIntern Week 4  
**Domain:** Machine Learning and Logistics