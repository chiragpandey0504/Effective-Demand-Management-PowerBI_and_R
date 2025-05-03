# Data-Driven Forecasting for Effective Demand Management in the E-Commerce Ecosystem

**Author:** Chirag Pandey  
**Degree:** MS in Information Systems & Technology (Business Intelligence), CSU San Bernardino  
**Date:** May 2024  

---

## Table of Contents
1. [Introduction & Publication](#introduction--publication)  
2. [Research Objectives](#research-objectives)  
3. [Datasets](#datasets)  
4. [Tools & Technologies](#tools--technologies)  
5. [Methodology](#methodology)  
6. [Key Findings](#key-findings)  
7. [Contact](#contact)  

---

## Introduction & Publication
In a rapidly evolving e-commerce landscape, accurately forecasting demand is critical to optimizing inventory, reducing stockouts, and maximizing profitability. This project investigates how **customer demographics** (age, gender, geography) and **product attributes** (category, brand) influence sales forecasting models—and compares traditional regression vs. modern machine-learning approaches.

> **For full methodology, results, and discussion, please refer to our publication:**  
> “Data-Driven Forecasting for Effective Demand Management in the E-Commerce Ecosystem”  
> <https://scholarworks.lib.csusb.edu/cgi/viewcontent.cgi?article=3119&context=etd>

---

## Research Objectives
- **Q1:** How do customer demographic variables (age, gender, location) impact e-commerce sales forecasting?  
- **Q2:** How do product-specific factors (category, brand) influence forecasting accuracy?  
- Evaluate and compare **Linear Regression** and **XGBoost** for predictive performance.

---

## Datasets
1. **Looker Ecommerce BigQuery Dataset** (orders, users, products, order_items, inventory, events, distribution_centers)  
   – Kaggle: <https://www.kaggle.com/datasets/mustafakeser4/looker-ecommerce-bigquery-dataset>  
2. **Worldometer GDP per Capita** (for country-level economic context)  
   – Worldometers: <https://www.worldometers.info/gdp/gdp-by-country/>

---

## Tools & Technologies
- **Power BI** for interactive dashboards and descriptive analytics  
- **R** (with packages `tidyverse`, `caret`, `xgboost`) for data preprocessing, modeling, and evaluation  

---

## Methodology

### Descriptive Analysis (Power BI)
1. Data cleaning & modeling (merging CSVs, filtering completed orders)  
2. Feature engineering via DAX (age groups, profit %, etc.)  
3. Visual exploration of:  
   - Sales by age, gender, country  
   - Top-performing states & customers  
   - Sales & profit by category and brand  

### Predictive Modeling (R)
1. Preprocess: factorize categoricals, remove outliers (IQR method)  
2. Split data: 70% train / 30% test  
3. Train **Multiple Linear Regression** & **XGBoost**  
4. Compare **R²** on train/test sets  
5. Extract feature importance from XGBoost  

---

## Key Findings
- **Demographics:**  
  - Age 55+ drives ~27% of sales in both USA & Brazil  
  - Male customers account for ~53% of sales  
- **Product Factors:**  
  - Top 10 categories (e.g., Outwear, Jeans) → 68% of sales  
  - Top 20 brands (e.g., Diesel, Calvin Klein) → 22% of sales  
- **Model Performance:**  
  - Linear Regression R² ≈ 0.97  
  - **XGBoost** R² ≈ 0.99 (train) / 0.99 (test) — outperforms regression  

---

## 📞 Contact

**Chirag Pandey**  
– Email: chiragpandey0504@gmail.com  
– GitHub: [@chiragpandey0504](https://github.com/chiragpandey0504)
