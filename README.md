### README for Stock Price Prediction Project  

---

# **📈 Stock Price Prediction Using Sentiment Analysis**  

This project leverages public sentiment from Reddit discussions and historical stock price data to predict stock movements for major companies like Tesla, Apple, and Amazon. Designed for ease of use in Google Colab, this project offers a seamless, end-to-end pipeline for data collection, preprocessing, sentiment analysis, and stock price prediction.  

---

## **🌟 Highlights**  

- **🚀 End-to-End Workflow**: A single Colab notebook encapsulating data collection, cleaning, analysis, and prediction.  
- **📊 Sentiment Analysis**: Analyze Reddit discussions using FinBERT, a sentiment analysis model tailored for financial data.  
- **💹 Stock Price Data**: Combine public sentiment with historical stock price trends to uncover actionable insights.  
- **🤖 Machine Learning Models**: Train and evaluate multiple machine learning algorithms to predict stock price movements effectively.  

---

## **🔧 Repository Structure**

```  
📂 stock_trends_prediction/  
├── stock_price_prediction.ipynb   # The main Colab notebook  
├── data/                          # Directory for datasets  
│   ├── stock_data_raw.csv         # Raw Reddit data  
│   ├── stock_cleaned.csv          # Cleaned Reddit data  
│   ├── stock_preprocessed.csv     # Preprocessed sentiment data  
│   ├── all_companies_classification_data.csv  # Stock data with labels  
│   └── merged_stock_sentiment_data.csv        # Final merged dataset  
├── results/                       # Directory for outputs  
│   └── results.txt                # Final result matrix (model evaluation)  
└── README.md                      # Project documentation  
```  

---

## **🛠 Workflow Overview**

### **1️⃣ Data Collection**  
- **Reddit Posts**: Use the Reddit API to scrape discussions from subreddits like `r/stocks` and `r/wallstreetbets`.  
- **Stock Data**: Fetch historical stock prices for Tesla, Apple, and Amazon using Yahoo Finance.  

### **2️⃣ Data Preprocessing**  
- **Text Cleaning**: Remove duplicates, special characters, and stopwords from Reddit posts.  
- **Sentiment Analysis**: Use FinBERT to classify each post as **Positive**, **Neutral**, or **Negative**.  
- **Feature Engineering**: Merge sentiment data with stock price data by aligning dates and companies.  

### **3️⃣ Model Training**  
- Train and evaluate the following machine learning models:  
  - **Logistic Regression**  
  - **Random Forest**  
  - **Gradient Boosting**  
  - **Support Vector Machines (SVM)**  
  - **LightGBM**  

- Use **GridSearchCV** for hyperparameter tuning and evaluate models based on accuracy, precision, recall, and F1-score.  

### **4️⃣ Final Output**  
- **Best Model**: Display the best-performing model along with its configuration and metrics.  
- **Predictions**: Predict stock price movement as Increase, Decrease, or No Change.  

---

## **📂 Data Section**

### **Datasets**  
- **Raw Reddit Data**: Scraped discussions, stored in `stock_data_raw.csv`.  
- **Cleaned Reddit Data**: Processed data after cleaning and sentiment analysis (`stock_cleaned.csv` and `stock_preprocessed.csv`).  
- **Stock Data with Labels**: Historical stock prices annotated with target labels (`all_companies_classification_data.csv`).  
- **Merged Dataset**: Combined sentiment and stock data ready for model training (`merged_stock_sentiment_data.csv`).  

---

📂 **Data Access**  
Since the datasets are large and exceed GitHub's file upload limits, you can download the datasets from the following link:  

- [Download Datasets from Google Drive](https://drive.google.com/drive/folders/1clJ2-vsdypKNe32euxH6Rew2npiOpEGO?usp=sharing)


---

## **📁 Results Section**  

### **results.txt**  
The `results.txt` file summarizes the final evaluation metrics of the best-performing model, including:  
- **Model Name**  
- **Hyperparameters**  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  

---

## **🌟 How to Use**  

### **Step 1: Download the Notebook**  
- Save the `stock_price_prediction.ipynb` file to your **local system** or **Google Drive**.  

### **Step 2: Open in Google Colab**  
- Upload the notebook to **Google Colab** or open it directly from your Google Drive.  

### **Step 3: Install Required Libraries**  
Run the following command in Colab to set up the environment:  
```python  
!pip install praw transformers torch pandas scikit-learn yfinance matplotlib lightgbm  
```  

### **Step 4: Add Reddit API Credentials**  
To scrape Reddit data, generate your own API keys:  
1. **Create a Reddit App**:  
   - Go to [Reddit's App Preferences](https://www.reddit.com/prefs/apps).  
   - Create a new **Script** app and note the **Client ID** and **Client Secret**.  
2. **Update Credentials**:  
   - Replace placeholders in the notebook with your credentials:  
     ```python  
     reddit = praw.Reddit(
         client_id="YOUR_CLIENT_ID",          
         client_secret="YOUR_CLIENT_SECRET",  
         user_agent="YOUR_USER_AGENT"         
     )
     ```  

### **Step 5: Run the Notebook**  
- Execute each cell in sequence to:  
  1. Collect data.  
  2. Preprocess text and stock data.  
  3. Perform sentiment analysis.  
  4. Train models and evaluate performance.  

### **Step 6: View Results**  
- **Model Performance**: Check the `results/results.txt` file for metrics.  
- **Processed Data**: Access datasets in the `data/` folder.  

---

## **📚 Key Insights**

- **Sentiment Distribution**: Analyze how public sentiment correlates with stock price movement.  
- **Model Performance**: Identify the best model for reliable stock trend prediction.  
- **Actionable Predictions**: Leverage insights for informed trading decisions.  

---

## **🤝 Contributors**  

- **Abhinav**: Developer and Data Scientist  

---

## **💬 Feedback**  

We welcome your contributions and suggestions to improve this project. Let’s collaborate to make stock price prediction smarter and more accurate! 😊  
