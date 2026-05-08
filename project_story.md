# Project Story

I built an AI-assisted predictive maintenance optimizer for industrial chemical reactors.

The goal was to move beyond simple prediction and create a decision-support system. The project uses synthetic chemical process monitoring data with multiple reactors, operating regimes, sensor readings, fault labels, efficiency loss, and time-to-fault information.

First, I cleaned and prepared the data using Python and pandas. I handled missing sensor values, created a binary failure-risk target, added time-based features, and generated rolling-window sensor features to capture gradual fault development.

Second, I trained machine learning models to predict failure risk. I compared Logistic Regression as a baseline with Random Forest as a nonlinear model. Because fault data is imbalanced, I evaluated the models using precision, recall, F1-score, ROC-AUC, threshold analysis, and leave-one-reactor-out validation.

Third, I converted predicted failure probabilities into reactor-level maintenance inputs. For each reactor, I calculated predicted failure risk, efficiency loss, and urgency based on time-to-fault.

Finally, I built a binary optimization model using PuLP. The model selects which reactors should receive preventive maintenance while respecting technician-hour, downtime, and budget constraints.

The final output is a maintenance recommendation, not just a prediction. This demonstrates how machine learning and optimization can be combined to support industrial decision-making.
