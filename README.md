# **Project Report: Customer Segmentation using K-Means Clustering**

## **1. Introduction**
Customer segmentation is a crucial strategy for businesses to **understand their customer base, tailor marketing strategies, and improve customer satisfaction**. This project applies **K-Means Clustering**, an **unsupervised machine learning algorithm**, to segment customers based on their purchasing behavior.

---

## **2. Objective**
The goal of this project is to:
- Segment customers into distinct groups based on their **annual income** and **spending score**.
- Identify potential customer clusters for **targeted marketing strategies**.
- Provide a **visual representation** of customer segments.

---

## **3. Dataset Overview**
### **3.1 Data Source**
The dataset used in this project is obtained from **GitHub**:  
📂 [Mall_Customers.csv](https://github.com/sangambhamare/Customer-Segmentation-using-K-Means-Clustering/blob/master/Mall_Customers.csv)

### **3.2 Data Description**
The dataset consists of **200 records** with the following attributes:
- **CustomerID**: Unique customer identifier (Removed for clustering)
- **Gender**: Gender of the customer (Removed for clustering)
- **Age**: Age of the customer
- **Annual Income (k$)**: Annual income of the customer in thousand dollars
- **Spending Score (1-100)**: A measure of the customer's spending behavior

### **3.3 Data Preprocessing**
- **Removed non-numeric attributes** (`CustomerID`, `Gender`).
- **Standardized features** (`Annual Income` and `Spending Score`) using **StandardScaler**.
- **Checked for missing values** and ensured data integrity.

---

## **4. Methodology**
### **4.1 K-Means Clustering**
K-Means is a popular **unsupervised learning algorithm** used for clustering. It groups similar data points together by minimizing intra-cluster distance.

### **4.2 Finding Optimal Clusters (K)**
To determine the optimal number of clusters (**K**), we used:
- **Elbow Method**: Measures intra-cluster variance (distortion) and identifies the best K.
- **Silhouette Score (Optional)**: Measures how well-separated the clusters are.

### **4.3 Steps Followed**
1. **Fetched Dataset** from GitHub using Python.
2. **Performed Data Cleaning & Feature Scaling**.
3. **Determined Optimal K** using the Elbow Method.
4. **Applied K-Means Algorithm** to segment customers.
5. **Visualized the clusters** using a scatter plot.

---

## **5. Results & Analysis**
### **5.1 Optimal Number of Clusters**
- Based on the **Elbow Method**, the optimal **number of clusters (K) = 5**.

### **5.2 Cluster Interpretations**
Each cluster represents a **customer group with similar purchasing behavior**:
1. **High Income, High Spending (Luxury Shoppers)**
2. **High Income, Low Spending (Frugal Customers)**
3. **Low Income, High Spending (Impulsive Shoppers)**
4. **Low Income, Low Spending (Budget Shoppers)**
5. **Middle Income, Moderate Spending (Average Shoppers)**

### **5.3 Visualization**
- The customer segments are visualized in a **scatter plot** where:
  - **X-Axis**: Annual Income (k$)
  - **Y-Axis**: Spending Score (1-100)
  - **Color**: Represents the clusters

- 📊 **Findings**:
  - **Luxury shoppers (high spending, high income)** may be **targeted for premium offers**.
  - **Frugal customers (high income, low spending)** may need **loyalty programs** to boost spending.
  - **Impulsive shoppers (low income, high spending)** may benefit from **discount-based promotions**.

---

## **6. Implementation**
### **6.1 Technologies Used**
- **Python**
- **Streamlit** (for building an interactive web app)
- **Scikit-Learn** (for K-Means clustering)
- **Matplotlib & Seaborn** (for data visualization)
- **Pandas & NumPy** (for data manipulation)
- **Requests** (to fetch dataset from GitHub)

### **6.2 Streamlit Web App**
A **Streamlit-based web application** was developed to:
✅ **Fetch the dataset** directly from GitHub  
✅ **Allow users to select the number of clusters (K)**  
✅ **Visualize customer segments** in an interactive scatter plot  
✅ **Provide a downloadable segmented dataset**  

---

## **7. Conclusion**
### **7.1 Key Takeaways**
- **K-Means Clustering effectively segments customers** based on spending patterns and income levels.
- Businesses can **use these insights to personalize marketing strategies** for different customer segments.
- **The Streamlit web app** makes the analysis **interactive and accessible**.

### **7.2 Future Enhancements**
🔹 Use **Hierarchical Clustering** for alternative segmentation  
🔹 Incorporate **DBSCAN** to detect outliers  
🔹 Add **real-time customer data integration** for dynamic segmentation  

---

## **8. References**
1. Kaggle Dataset: Customer Segmentation  
2. Scikit-Learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)  
3. Streamlit Documentation: [https://streamlit.io](https://streamlit.io)  

---

# **🎯 Final Thoughts**
This project successfully segments customers using **K-Means Clustering** and provides a **fully functional Streamlit web app** for interactive analysis. 🚀
