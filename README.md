# Movie-Recommendation-System
# 🎬 AI/ML MINI PROJECT – Movie Recommender System (Flask + KNN)

## 📌 Project Overview

This mini-project is a **Personalized Movie Recommender System** built using **Machine Learning (KNN algorithm)** and deployed using **Flask**.
The system recommends movies to the user based on **similar genres** and **movie similarity scores**.

The project uses:

* **movies.csv** (Indian movies dataset – Bollywood + South Indian Hindi-dubbed)
* **KNN similarity model**
* **Flask Web Interface**
* **Preprocessing Module**
* **Graph/Visualization Support**

---

## 📁 Project Structure

```
AIML MINI PROJECT/
│── templates/
│     └── index.html           # Frontend page
│
│── app.py                     # Flask web application
│── knn_model.py               # KNN model + similarity logic
│── preprocess.py              # Dataset loading + preprocessing
│── movies.csv                 # Movie dataset (Bollywood + South Indian)
│── README.md                  # Documentation
```

---

## 🧠 Concepts Used (AIML Concepts)

### **1. Supervised ML – KNN (K-Nearest Neighbors)**

* Used to compute **similarity between movies**.
* Feature vectors created using **genre encoding (one-hot)**.
* KNN finds movies with closest feature similarity.

### **2. Data Preprocessing**

* Loading dataset
* Cleaning missing values
* Converting `genres` into ML-friendly form
* One-hot encoding (binary matrix)
* Normalization

### **3. Feature Engineering**

* Genre transformation → numerical vectors
* Movie similarity matrix
* Optional dimensionality reduction using PCA

### **4. Visualization (Optional)**

* Plotting similarity score bars
* Matplotlib used for graphs

### **5. Flask Web Application**

* GET and POST request handling
* Accepts movie input from user
* Displays recommended movie list
* Renders templates using Jinja2

---

## ⚙️ Installation

### **1. Install Dependencies**

```bash
pip install flask pandas scikit-learn matplotlib
```

---

## ▶️ Running the Project

### **Step 1: Place all files in one folder**

```
AIML MINI PROJECT/
```

### **Step 2: Start Flask Server**

```bash
python app.py
```

### **Step 3: Open Browser**

```
http://127.0.0.1:5000/
```

You will see the movie recommender homepage.

---

## 🎯 Features

✔ Movie search-based recommendation
✔ Indian movies dataset (Bollywood + South Indian)
✔ ML-based similarity scoring
✔ Simple HTML interface
✔ Easily extendable
✔ Graph support
✔ Ready for mini-project submission

---

## 📊 Dataset Used – `movies.csv`

Contains:

* **movieId** – unique ID
* **title** – movie name
* **genres** – genres separated by `|`

Example:

```
movieId,title,genres
1,Harry Potter,Magic|Drama
```

Dataset size: **1000 movies** 

---

## 🧪 How Movie Recommendation Works

1. User enters movie name
2. System finds the movie in dataset
3. Convert genres → numeric vector
4. KNN finds closest matching movies
5. Sort by similarity
6. Display top recommended movies

---

## 🧩 Future Improvements

* Add user rating matrix
* Cosine similarity instead of KNN
* Add images, posters
* Add multiple filtering options
* Deploy on cloud (Heroku / Render / AWS)


