# 📊 Tesla Price Prediction

A machine learning project that predicts future stock price movements using Logistic Regression, Support Vector Classifier (SVC), and XGBoost. This project involves feature engineering, data visualization, and performance evaluation of the models.

## 📁 Project Structure

```
Stock-Price-Prediction/
├── Dataset/
│    └── Tesla.csv
├── main.py
└── README.md
```

## 📌 Features

- Analyzes Tesla stock data with various visualizations
- Performs feature engineering for better prediction
- Implements Logistic Regression, SVC (poly kernel), and XGBoost classifiers
- Evaluates models using AUC-ROC, precision, recall, and F1-score
- Visualizes performance through confusion matrices and correlation heatmaps

## 📊 Dataset

The dataset (`Tesla.csv`) contains the following columns:
- **Date**: Date of the stock data
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Adj Close**: Adjusted closing price
- **Volume**: Trading volume

## 📦 Dependencies

Ensure you have Python installed along with the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## 🚀 Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

2. Ensure the dataset is placed in the `Dataset/` folder.

3. Run the main script:

```bash
python main.py
```

## 📈 Model Evaluation Metrics

The script outputs the following metrics for each model:

- **AUC-ROC** (Area Under the Curve - Receiver Operating Characteristic)
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

## 📊 Visualizations

The project includes the following visualizations:

- Tesla's closing price over time
- Distribution plots of key financial features
- Box plots to detect outliers
- Yearly average stock data
- Correlation heatmap
- Target variable distribution (Next-day price movement)
- Confusion matrix for each model

### Example Output:

1. **Stock Closing Price Over Time**

![Closing Price](images/closing_price.png)

2. **Confusion Matrix**

![Confusion Matrix](images/confusion_matrix.png)

## 🧠 Models Used

1. **Logistic Regression**
2. **Support Vector Classifier (SVC)** (with polynomial kernel)
3. **XGBoost Classifier**

## 📚 Insights

- Feature engineering (e.g., `open-close`, `low-high`, and `is_quarter_end`) improves prediction performance.
- XGBoost tends to perform better due to its ability to capture complex relationships.
- Proper data normalization enhances SVM and Logistic Regression accuracy.

## 📝 Future Improvements

- Incorporate LSTM models for time-series forecasting
- Expand dataset to other companies
- Fine-tune hyperparameters for better model performance

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

### 🌟 Star the repo if you found it useful! 🌟

