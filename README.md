# 🌟 Cryptocurrency Volatility Prediction and Portfolio Strategy 📈

Welcome to our project **"crypto-volatility-lab"**, based on the paper *"Volatility Forecasting with Neural Networks for Portfolio Optimization of Cryptocurrencies Indexes"* which can be found in this repository.

## 🎯 Goals and Outcomes

- Build an **end-to-end cryptocurrency analysis pipeline**.
- Leverage **AI and neural networks** to predict market behavior.
- Construct and compare **portfolio allocation strategies** to maximize returns and manage risk effectively.
- Develop a **modular, package-oriented approach** in Python for seamless usability and scalability.

## 🛠 Installation

To set up the environment and install dependencies, follow these steps:

#### 1️⃣ Create a Virtual Environment

It is recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 2️⃣ Install Dependencies

After activating the virtual environment, install the required dependencies and the crypto-volatility-lab package!:

```bash
pip install -r requirements.txt
pip install -e .
```
---

## 📊 See All the Analysis and Results

To explore the entire study pipeline, analysis, and results, you can check out the following Jupyter notebook:

## 📂 sandbox/Notebook_Final_Projet.ipynb

This notebook provides insights into data preprocessing, model training, performance evaluation, and portfolio strategy comparisons.


## 🚀 Running the API
Our API provides an interface for accessing the scraping, time series and features creation, volatility prediction models and portfolio optimization objects.
Once the setup is complete, you can start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

## 📂 Project Structure

```
crypto-volatility-lab/
│-- api/
    ├── models/
    ├── templates/
    ├── main.py
│-- crypto_volatility_lab/
│   ├── data_construction/
│   │   ├── cryptoScraper.py
│   │   ├── featuresCreator.py
│   │   ├── timeSeriesCreator.py
│   ├── modeling/
│   │   ├── gruPipeline.py
│   │   ├── lstmgruPipeline.py
│   │   ├── lstmPipeline.py
│   │   ├── modelPipeline.py
│   │   ├── tcnnPipeline.py
│   ├── portfolio_optimization/
        ├── portfolioConstructor.py
│-- sandbox/
```

## 👥 Authors

- **[Lina Benzemma](https://github.com/linabnz)**
- **[Sharon Chemmama](https://github.com/Sharon2607)**
- **[Emma Eberle](https://github.com/emmaebrl)**


---

