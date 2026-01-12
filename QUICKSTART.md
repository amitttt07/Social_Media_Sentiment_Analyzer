# 🚀 Quick Start Guide

Get up and running with the Sentiment Analyzer in just 5 minutes!

---

## ⚡ Super Quick Start (For the Impatient)

```bash
# 1. Install packages
pip install streamlit pandas numpy scikit-learn nltk plotly wordcloud matplotlib seaborn openpyxl

# 2. Run the app
streamlit run app.py

# 3. Click "Create Sample Dataset" in sidebar
# 4. Upload the generated file
# 5. Click "Train Model"
# 6. Start analyzing! 🎉
```

---

## 📋 Step-by-Step Guide

### Step 1: Setup (2 minutes)

**Open your terminal/command prompt and run:**

```bash
# Navigate to project folder
cd sentiment_analyzer

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install everything at once
pip install -r requirements.txt
```

### Step 2: Run the App (30 seconds)

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

### Step 3: Get a Dataset (Choose one)

#### Option A: Use Sample Data (Fastest - 10 seconds)
1. Look at the **sidebar** (left side of the screen)
2. Click **"📝 Create Sample Dataset for Testing"**
3. File `sample_dataset.csv` will be created
4. You're ready to train!

#### Option B: Download Real Data (2 minutes)
1. Go to [Twitter Airline Sentiment on Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
2. Click "Download" (you may need to login/signup)
3. Extract the ZIP file
4. Find `Tweets.csv`

### Step 4: Train the Model (2-5 minutes)

1. In the **sidebar**, click **"Browse files"** under "Airline Sentiment CSV/Excel"
2. Select your CSV file (`sample_dataset.csv` or `Tweets.csv`)
3. You'll see a preview of your data
4. Set **"Samples per class"** to **1000** (for quick testing)
5. Click **"🚀 Train Model"**
6. Wait for training to complete (you'll see progress bars)
7. Look for **"✅ Model trained successfully!"** and balloons! 🎈

### Step 5: Analyze Text (30 seconds)

1. Go to **"🎯 Analyze Text"** tab
2. Type something like: `"This is amazing! Best day ever!"`
3. Click **"🔍 Analyze Sentiment"**
4. See the results with emoji, sentiment label, and confidence!

---

## 🎯 What to Try Next

### Try Different Texts

**Positive:**
```
"I absolutely love this product! It's fantastic!"
"Best service I've ever experienced!"
"Amazing quality, highly recommend!"
```

**Neutral:**
```
"It's okay, nothing special."
"Average product, meets expectations."
"Decent for the price."
```

**Negative:**
```
"Terrible experience, very disappointed!"
"Worst purchase ever, waste of money!"
"Horrible quality, don't buy!"
```

### Try Batch Analysis

1. Create a file `test.csv`:
```csv
text
"Great product!"
"Not good."
"It's fine."
"Amazing!"
"Terrible."
```

2. Go to **"📊 Batch Analysis"** tab
3. Upload `test.csv`
4. Click **"🚀 Analyze All"**
5. Download the results!

### Explore Visualizations

- **📈 Model Insights** - See accuracy and performance metrics
- **☁️ Word Clouds** - Visual word frequency by sentiment
- **📚 Dataset Info** - Understand your training data

---

## 🆘 Common First-Time Issues

### Issue 1: "Module not found"
**Fix:**
```bash
pip install [missing_module]
# Or reinstall everything:
pip install -r requirements.txt
```

### Issue 2: "Port 8501 is already in use"
**Fix:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Issue 3: "Only one class" error
**Fix:**
- Your CSV needs **multiple sentiment types** (Positive, Neutral, Negative)
- Use the **"Create Sample Dataset"** button for a working example
- Check your CSV has a `sentiment` column with different values

### Issue 4: Training takes forever
**Fix:**
- Reduce "Samples per class" to 500-1000
- Close other applications
- Be patient (first time can take 3-5 minutes)

### Issue 5: NLTK data not found
**Fix:**
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 💡 Pro Tips

1. **Start Small**: Use 500-1000 samples per class for your first training
2. **Sample Data First**: Always test with the built-in sample generator first
3. **Check Preview**: Look at the data preview before training
4. **Read Messages**: The app shows helpful error messages - read them!
5. **Be Patient**: First training takes 2-5 minutes, it's normal

---

## 🎓 Next Steps

Once you've got the basics working:

1. **Try Larger Datasets**: Increase to 3000-5000 samples per class
2. **Combine Multiple Datasets**: Upload 2-3 different datasets
3. **Experiment**: Try analyzing your own social media posts
4. **Customize**: Modify the code to add new features
5. **Share**: Show it to friends and get their feedback!

---

## 📞 Need Help?

**Stuck? Don't worry!**

1. Check the main README.md for detailed troubleshooting
2. Look for error messages in the terminal
3. Try the sample dataset first
4. Open an issue on GitHub with screenshots

---

## ✅ Success Checklist

After following this guide, you should have:

```
☐ Installed all required packages
☐ Run the app successfully
☐ Created or uploaded a dataset
☐ Trained a model
☐ Analyzed at least one text
☐ Seen the balloons animation 🎈
☐ Explored different tabs
☐ Downloaded batch results (optional)
```

If you checked all boxes - **Congratulations!** 🎉

You're now ready to use the Sentiment Analyzer for real projects!

---

## 🚀 Ready for More?

Check out:
- **README.md** - Full documentation
- **Code comments** - Learn how it works
- **Kaggle datasets** - More data to try
- **Customization** - Make it your own

---

**Time to completion**: ~5-10 minutes
**Difficulty**: Beginner-friendly
**Support**: Full documentation available

**Happy Analyzing!** 🎯