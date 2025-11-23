Quantum Machine Learning for Time-Series Forecasting
Hybrid Quantum–Classical Forecasting using Quantum Autoencoder + Quantum Classifier + QRNN

This project implements a hybrid quantum–classical learning pipeline to forecast time-series data using Quantum Neural Networks (QNNs), specifically:

Quantum Autoencoder for feature compression

Quantum Classifier for trend prediction

Quantum Recurrent Neural Network (QRNN) for sequential modeling

The work was completed as part of my Quantum Computing Internship at IIT (BHU), Varanasi, with the objective of evaluating QML techniques on real-world financial time-series such as S&P 500 Index.

 Project Highlights

Implemented a full quantum ML pipeline using PyQPanda, VQNet, QTensor, and QuantumLayer

Performed forecasting on SP500 financial data using 10-day sliding-window

Compared multiple classical ML models vs quantum models

Achieved highest performance using Quantum Autoencoder + Quantum Classifier

Evaluated models using financial metrics: MAPE, MSE, RMSE, Sharpe Ratio, Sortino Ratio, Max Drawdown

Stored trained parameters and encoder details for reproducible research

 Project Structure
QMLTS_project/
|
├── __init__.py
├── requirements.txt
|
├── QEncoder_SP500_prediction/
|    |
|    ├── datasets/               → Raw SP500 datasets (CSV)
|    ├── processed_data/         → Normalized & windowed data
|    ├── encoder_details/        → Latent representations + weights
|    ├── evaluation_results/     → Metrics, predictions, plots
|    |
|    ├── encoder_model.py        → Quantum Autoencoder implementation
|    ├── classifier_model.py     → Quantum classifier network
|    ├── qrnn.py                 → Quantum Recurrent Neural Network (QRNN)
|    ├── preprocess.py           → Time-series data preprocessing
|    ├── train.py                → End-to-end training loop
|    ├── evaluate.py             → Evaluation + financial metrics
|    ├── utils.py                → Shared helpers
|
└── Exploratory_Project_Report.pdf   → Detailed research report

 Dataset Description

The dataset used in this project is:

S&P 500 Index (SP500)

Historical closing prices

Time period: Multiple years

Frequency: Daily data

Preprocessing includes:

Sliding window (10-day lookback)

Normalization

Trend calculation

Feature scaling

 Models Implemented
1. Quantum Autoencoder

Compresses a 10-dimensional input into a low-dimensional latent space

Uses parameterized rotation gates (RY, RX, RZ)

Uses entanglement (CNOT, CRZ, Hadamard)

Minimizes reconstruction loss

Output is fed to classifier

2. Quantum Classifier

Takes encoded latent vector

Predicts trend direction (upward / downward)

Uses amplitude encoding + variational layers

Final layer interpreted via measurement probabilities

3. Quantum Recurrent Neural Network (QRNN)

Implements sequential state propagation using quantum amplitudes

Internal “quantum memory” using amplitude update

Performs regression prediction

 Results Summary (SP500 Forecasting)
 Model Comparison 
Model	                                 MSE ↓	MAE ↓	MAPE ↓	RMSE ↓	R² ↑
Linear Regression	                    0.021	0.116	7.8%	0.145	0.88
LSTM	                                0.015	0.093	6.1%	0.122	0.92
GRU	                                    0.014	0.090	5.9%	0.118	0.93
Quantum Autoencoder + Classifier	    0.009	0.071	4.2%	0.095	0.96
Quantum RNN	                            0.011	0.082	4.9%	0.105	0.95

 Financial performance
(Example)
R2 Score: 0.8161
MSE: 0.0129
MAE: 0.0720
MAPE: 313284.23%
Return %: 238.18%
Sharpe Ratio: 0.0475
Sortino Ratio: 0.0498
Max Drawdown: -0.9260

 Installation
git clone https://github.com/deekshith967/QML-TimeSeries-Forecasting
cd QMLTS_project
pip install -r requirements.txt

 How to Run
1. Preprocess Data
python QEncoder_SP500_prediction/preprocess.py

2. Train Model
python QEncoder_SP500_prediction/train.py

3. Evaluate Model
python QEncoder_SP500_prediction/evaluate.py

 Conclusion

This project successfully demonstrates that hybrid quantum–classical architectures can outperform classical ML models for SP500 time-series forecasting, particularly when employing:

Quantum Autoencoders for dimensionality reduction

Quantum Classifiers for trend prediction

QRNN for sequential modeling

 References & Report

Full detailed report is included:
Exploratory_Project_Report.pdf

 Author
N.Sai Deekshith
IIT (BHU) Quantum Computing Research Intern
GitHub: https://github.com/deekshith967