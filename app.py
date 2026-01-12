import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="✨ Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #E0C3FC 0%, #8EC5FC 100%);
    }
    
    h1 {
        background: linear-gradient(120deg, #a855f7 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    
    h2, h3 {
        color: #6b21a8;
        font-weight: 600;
    }
    
    .stButton>button {
        background: linear-gradient(120deg, #a855f7 0%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(168, 85, 247, 0.4);
    }
    
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e9d5ff;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #a855f7;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6b21a8;
        font-weight: 500;
        margin-top: 3rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Text preprocessing functions
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                                   'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him'])
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

# Data loading and preparation functions
class DatasetLoader:
    """Load and prepare datasets from multiple sources"""
    
    @staticmethod
    def load_amazon_reviews(file_path):
        """Load Amazon reviews dataset"""
        try:
            df = None
            
            # Check if it's an Excel file or CSV
            if hasattr(file_path, 'name'):
                file_name = file_path.name
            else:
                file_name = str(file_path)
            
            # Read Excel or CSV
            if file_name.endswith(('.xlsx', '.xls')):
                st.info("📊 Detected Excel file, reading...")
                try:
                    df = pd.read_excel(file_path)
                    st.success(f"✅ Excel file loaded: {len(df)} rows")
                except Exception as e:
                    st.error(f"Error reading Excel: {e}")
                    return None
            else:
                # Try reading with different encodings for CSV
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except:
                        continue
            
            if df is None:
                st.error("Could not read file with any encoding")
                return None
            
            st.info(f"Amazon dataset columns: {list(df.columns)}")
            
            # Amazon dataset typically has 'review_text' and 'rating' columns
            if 'review_text' in df.columns and 'rating' in df.columns:
                df = df.rename(columns={'review_text': 'text'})
                # Convert ratings to sentiment (1-2: Negative, 3: Neutral, 4-5: Positive)
                df['sentiment'] = df['rating'].apply(lambda x: 
                    'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral'))
                return df[['text', 'sentiment']]
            
            # Alternative column names
            if 'review' in df.columns:
                df = df.rename(columns={'review': 'text'})
            
            # Check for label column (binary format)
            if 'label' in df.columns and 'text' in df.columns:
                df['sentiment'] = df['label'].apply(lambda x: 
                    'Positive' if x == 2 or x == 1 else 'Negative')
                return df[['text', 'sentiment']]
            
            # If already has text and sentiment
            if 'text' in df.columns and 'sentiment' in df.columns:
                return df[['text', 'sentiment']]
            
            st.warning(f"Unrecognized Amazon dataset format. Columns: {list(df.columns)}")
            return None
        except Exception as e:
            st.error(f"Error loading Amazon reviews: {e}")
            return None
    
    @staticmethod
    def load_sentiment140(file_path):
        """Load Sentiment140 Twitter dataset"""
        try:
            df = None
            
            # Check if it's an Excel file or CSV
            if hasattr(file_path, 'name'):
                file_name = file_path.name
            else:
                file_name = str(file_path)
            
            # Read Excel or CSV
            if file_name.endswith(('.xlsx', '.xls')):
                st.info("📊 Detected Excel file, reading...")
                try:
                    df = pd.read_excel(file_path)
                    st.success(f"✅ Excel file loaded: {len(df)} rows")
                    
                    # If it has headers, use them
                    if 'target' not in df.columns and len(df.columns) >= 6:
                        df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
                except Exception as e:
                    st.error(f"Error reading Excel: {e}")
                    return None
            else:
                # Sentiment140 has columns: target, ids, date, flag, user, text
                try:
                    df = pd.read_csv(file_path, encoding='latin-1', header=None,
                                   names=['target', 'ids', 'date', 'flag', 'user', 'text'])
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    return None
            
            if df is None:
                st.error("Could not read file")
                return None
            
            # Convert target (0=negative, 2=neutral, 4=positive)
            df['sentiment'] = df['target'].apply(lambda x: 
                'Positive' if x == 4 else ('Negative' if x == 0 else 'Neutral'))
            
            return df[['text', 'sentiment']]
        except Exception as e:
            st.error(f"Error loading Sentiment140: {e}")
            return None
    
    @staticmethod
    def load_airline_sentiment(file_path):
        """Load Twitter Airline Sentiment dataset"""
        try:
            df = None
            
            # Check if it's an Excel file or CSV
            if hasattr(file_path, 'name'):
                file_name = file_path.name
            else:
                file_name = str(file_path)
            
            # Read Excel or CSV
            if file_name.endswith(('.xlsx', '.xls')):
                st.info("📊 Detected Excel file, reading...")
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    st.success(f"✅ Excel file loaded: {len(df)} rows")
                except Exception as e:
                    st.error(f"Error reading Excel: {e}")
                    return None
            else:
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        st.success(f"✅ CSV file loaded: {len(df)} rows")
                        break
                    except:
                        continue
            
            if df is None:
                st.error("Could not read file with any method")
                return None
            
            # Debug: Show ALL columns
            st.info(f"📋 All columns found: {list(df.columns)}")
            st.info(f"📊 Dataset shape: {df.shape}")
            
            # Debug: Show first few rows
            st.info("🔍 First 3 rows preview:")
            st.dataframe(df.head(3))
            
            # Check for airline_sentiment column
            if 'airline_sentiment' in df.columns:
                st.success("✅ Found 'airline_sentiment' column!")
                
                # Check unique values BEFORE processing
                unique_sentiments = df['airline_sentiment'].unique()
                st.info(f"🎯 Unique sentiment values in file: {unique_sentiments}")
                
                # Show distribution BEFORE capitalization
                st.info(f"📊 Raw sentiment distribution:\n{df['airline_sentiment'].value_counts()}")
                
                # Now rename and capitalize
                df = df.rename(columns={'airline_sentiment': 'sentiment'})
                df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
                
                # Show distribution AFTER capitalization
                st.info(f"✅ After capitalization:\n{df['sentiment'].value_counts()}")
                
                # Return only text and sentiment columns
                result_df = df[['text', 'sentiment']].copy()
                
                # Final check
                st.success(f"🎉 Returning {len(result_df)} rows with sentiments: {result_df['sentiment'].unique()}")
                
                return result_df
            
            # Try alternative column names
            if 'sentiment' in df.columns and 'text' in df.columns:
                st.info("Found 'sentiment' and 'text' columns")
                df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
                return df[['text', 'sentiment']].copy()
            
            # Check for 'tweet_text' column
            if 'tweet_text' in df.columns:
                df = df.rename(columns={'tweet_text': 'text'})
                if 'airline_sentiment' in df.columns:
                    df = df.rename(columns={'airline_sentiment': 'sentiment'})
                    df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
                    return df[['text', 'sentiment']].copy()
            
            st.error(f"❌ Could not find required columns!")
            st.error(f"Available columns: {list(df.columns)}")
            st.info("Expected: 'text' and 'airline_sentiment' OR 'text' and 'sentiment'")
            return None
            
        except Exception as e:
            st.error(f"❌ Error loading Airline sentiment: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    @staticmethod
    def combine_and_balance_datasets(dataframes, samples_per_class=5000):
        """Combine multiple dataframes and balance classes"""
        if not dataframes:
            st.error("No valid datasets provided!")
            return None
        
        st.info(f"🔄 Combining {len(dataframes)} dataset(s)...")
        
        # Debug: Check each dataframe before combining
        for idx, df in enumerate(dataframes):
            st.info(f"Dataset {idx+1}: {len(df)} rows")
            st.info(f"  Sentiments: {df['sentiment'].unique()}")
            st.info(f"  Distribution: {df['sentiment'].value_counts().to_dict()}")
        
        # Concatenate all datasets
        combined = pd.concat(dataframes, ignore_index=True)
        st.info(f"📊 Combined dataset size: {len(combined)} samples")
        
        # Debug: Check combined data
        st.info(f"🎯 Combined sentiments: {combined['sentiment'].unique()}")
        st.info(f"📊 Combined distribution:\n{combined['sentiment'].value_counts()}")
        
        # Remove duplicates
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=['text'])
        after_dedup = len(combined)
        st.info(f"🔄 After removing duplicates: {after_dedup} samples (removed {before_dedup - after_dedup})")
        
        # Remove null values
        before_null = len(combined)
        combined = combined.dropna(subset=['text', 'sentiment'])
        after_null = len(combined)
        st.info(f"🔄 After removing nulls: {after_null} samples (removed {before_null - after_null})")
        
        # Check sentiment distribution again
        sentiment_counts = combined['sentiment'].value_counts()
        st.info(f"📊 Final sentiment distribution:\n{sentiment_counts}")
        
        # Ensure we have all three classes
        available_classes = set(combined['sentiment'].unique())
        st.info(f"✅ Available sentiment classes: {available_classes}")
        
        if len(available_classes) < 2:
            st.error(f"⚠️ Dataset only contains {len(available_classes)} sentiment class(es): {available_classes}")
            st.error("Please upload datasets with multiple sentiment classes (Positive, Neutral, Negative)")
            
            # Additional debugging
            st.error("🔍 Debugging information:")
            st.error(f"Total rows: {len(combined)}")
            st.error(f"Unique sentiments found: {combined['sentiment'].unique()}")
            st.error(f"Sample sentiment values:\n{combined['sentiment'].head(20)}")
            
            return None
        
        required_classes = ['Positive', 'Neutral', 'Negative']
        missing_classes = set(required_classes) - available_classes
        if missing_classes:
            st.warning(f"⚠️ Missing sentiment classes: {missing_classes}")
            st.info("The model will work but may have limited prediction capabilities.")
        
        # Balance classes
        min_samples = min(combined['sentiment'].value_counts().min(), samples_per_class)
        
        if min_samples < 100:
            st.error(f"⚠️ Not enough samples! Minimum class has only {min_samples} samples.")
            st.error("Please upload larger datasets or reduce 'Samples per class' setting.")
            return None
        
        balanced_dfs = []
        for sentiment in available_classes:
            sentiment_df = combined[combined['sentiment'] == sentiment]
            sample_size = min(len(sentiment_df), min_samples)
            if sample_size > 0:
                sentiment_df = sentiment_df.sample(n=sample_size, random_state=42)
                balanced_dfs.append(sentiment_df)
                st.success(f"✅ {sentiment}: {sample_size} samples selected")
        
        if not balanced_dfs:
            st.error("No valid data after balancing!")
            return None
        
        balanced = pd.concat(balanced_dfs, ignore_index=True)
        balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        st.success(f"🎉 Final balanced dataset: {len(balanced)} samples")
        st.info(f"Final distribution: {balanced['sentiment'].value_counts().to_dict()}")
        
        return balanced

# Model training function - DO NOT CACHE to ensure fresh training
def train_model(df, _preprocessor):
    """Train sentiment classification model"""
    
    if df is None or len(df) == 0:
        st.error("No data available for training!")
        return None, None, {}
    
    # Store original training samples count
    original_samples = len(df)
    
    # Check if we have multiple classes
    unique_sentiments = df['sentiment'].nunique()
    if unique_sentiments < 2:
        st.error(f"⚠️ Cannot train model with only {unique_sentiments} sentiment class!")
        st.error("Please upload datasets with at least 2 different sentiment classes.")
        return None, None, {}
    
    st.info("🔄 Training model... This may take a few minutes.")
    progress_bar = st.progress(0)
    
    # Preprocess text
    progress_bar.progress(20)
    st.info("Preprocessing text data...")
    df['cleaned_text'] = df['text'].apply(_preprocessor.clean_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    
    if len(df) < 100:
        st.error("⚠️ Not enough valid text data after preprocessing!")
        return None, None, {}
    
    st.success(f"✅ Preprocessed {len(df)} samples")
    progress_bar.progress(40)
    
    # Vectorize
    st.info("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    try:
        X = vectorizer.fit_transform(df['cleaned_text'])
        st.success(f"✅ Created {X.shape[1]} features")
    except Exception as e:
        st.error(f"Error during vectorization: {e}")
        return None, None, {}
    
    y = df['sentiment']
    
    progress_bar.progress(60)
    
    # Check class distribution before split
    class_counts = y.value_counts()
    st.info(f"📊 Class distribution: {dict(class_counts)}")
    
    # Only stratify if all classes have enough samples
    min_class_size = class_counts.min()
    test_size = 0.2
    
    if min_class_size < 2:
        st.error("⚠️ At least one class has fewer than 2 samples!")
        return None, None, {}
    
    # Adjust test_size if needed
    if min_class_size < 10:
        test_size = 0.1
        st.warning(f"Using smaller test size ({test_size}) due to limited data")
    
    # Split data with stratification
    st.info("Splitting train/test data...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        st.success(f"✅ Train size: {len(y_train)}, Test size: {len(y_test)}")
    except ValueError as e:
        st.error(f"Error splitting data: {e}")
        st.info("Trying without stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    progress_bar.progress(70)
    
    # Train model
    st.info("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs'
    )
    
    try:
        model.fit(X_train, y_train)
        st.success("✅ Model training complete!")
    except ValueError as e:
        st.error(f"Error training model: {e}")
        st.error("This usually means there's an issue with the class distribution in your data.")
        return None, None, {}
    
    progress_bar.progress(80)
    
    # Evaluate
    st.info("Evaluating model performance...")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # Get unique labels for confusion matrix
        unique_labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        
        st.success(f"✅ Model Accuracy: {accuracy:.2%}")
        st.success(f"✅ F1 Score: {f1:.2%}")
        
        progress_bar.progress(100)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'labels': unique_labels
        }
    except Exception as e:
        st.error(f"Error during evaluation: {e}")
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    
    # Save model with ALL necessary data
    model_data = {
        'vectorizer': vectorizer,
        'model': model,
        'preprocessor': _preprocessor,
        'metrics': metrics,
        'training_samples': original_samples,  # Use original count before preprocessing
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'classes': unique_labels
    }
    
    try:
        with open('sentiment_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        st.success("✅ Model saved successfully as 'sentiment_model.pkl'")
        st.info(f"📊 Training samples: {original_samples}")
        
        # Verify the save worked
        with open('sentiment_model.pkl', 'rb') as f:
            verify_data = pickle.load(f)
            verify_samples = verify_data.get('training_samples', 0)
            st.success(f"✅ Verification: Model file contains {verify_samples:,} training samples")
            
            if verify_samples != original_samples:
                st.warning(f"⚠️ Mismatch! Expected {original_samples}, got {verify_samples}")
    except Exception as e:
        st.error(f"❌ Could not save model: {e}")
        import traceback
        st.error(traceback.format_exc())
    
    return vectorizer, model, metrics

# Load or create model - DO NOT CACHE to ensure fresh reads
def load_or_train_model():
    """Load existing model or prompt for training"""
    try:
        if not os.path.exists('sentiment_model.pkl'):
            return None, None, TextPreprocessor(), {}, 0
        
        # Check if file is not empty
        file_size = os.path.getsize('sentiment_model.pkl')
        if file_size == 0:
            st.warning("⚠️ Model file is empty, deleting...")
            os.remove('sentiment_model.pkl')
            return None, None, TextPreprocessor(), {}, 0
        
        st.info(f"📦 Loading model from file ({file_size / 1024:.2f} KB)...")
        
        with open('sentiment_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Verify model_data has required keys
        required_keys = ['vectorizer', 'model', 'preprocessor']
        missing_keys = [key for key in required_keys if key not in model_data]
        
        if missing_keys:
            st.error(f"⚠️ Model file is corrupted (missing: {missing_keys})")
            st.info("Deleting corrupted model file...")
            os.remove('sentiment_model.pkl')
            return None, None, TextPreprocessor(), {}, 0
        
        training_samples = model_data.get('training_samples', 0)
        st.success(f"✅ Model loaded! Training samples: {training_samples:,}")
        
        return (
            model_data['vectorizer'],
            model_data['model'],
            model_data['preprocessor'],
            model_data.get('metrics', {}),
            training_samples
        )
    except EOFError:
        st.error("⚠️ Model file is corrupted (EOFError)")
        st.info("Deleting corrupted file...")
        try:
            os.remove('sentiment_model.pkl')
        except:
            pass
        return None, None, TextPreprocessor(), {}, 0
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("Attempting to delete corrupted file...")
        try:
            os.remove('sentiment_model.pkl')
        except:
            pass
        return None, None, TextPreprocessor(), {}, 0

# Helper functions
def get_sentiment_emoji(sentiment):
    emojis = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😠'}
    return emojis.get(sentiment, '🤔')

def get_sentiment_color(sentiment):
    colors = {'Positive': '#a855f7', 'Neutral': '#3b82f6', 'Negative': '#ec4899'}
    return colors.get(sentiment, '#94a3b8')

def predict_sentiment(text, vectorizer, model, preprocessor):
    """Predict sentiment for given text"""
    cleaned = preprocessor.clean_text(text)
    if not cleaned:
        return "Neutral", 50.0
    
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = max(probabilities) * 100
    
    return prediction, confidence

def create_wordcloud(texts, sentiment_type):
    """Generate word cloud for given texts"""
    combined_text = ' '.join(texts)
    
    if not combined_text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis' if sentiment_type == 'Positive' else ('coolwarm' if sentiment_type == 'Negative' else 'Blues'),
        max_words=50
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'{sentiment_type} Words', fontsize=16, fontweight='bold', 
                 color=get_sentiment_color(sentiment_type))
    
    return fig

# Main App
def main():
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'training_samples' not in st.session_state:
        st.session_state.training_samples = 0
    
    # Load model fresh every time (no caching)
    vectorizer, model, preprocessor, metrics, training_samples = load_or_train_model()
    
    # Sidebar for dataset management
    with st.sidebar:
        st.markdown("### 📦 Dataset Management")
        
        if not st.session_state.model_trained and vectorizer is None:
            st.warning("⚠️ No trained model found!")
            st.markdown("---")
            st.markdown("#### Upload Datasets")
            
            st.info("💡 **Tip:** You can also create a sample CSV file with 'text' and 'sentiment' columns to test quickly!")
            
            # Option to create sample dataset
            if st.button("📝 Create Sample Dataset for Testing", key='create_sample'):
                sample_data = {
                    'text': [
                        # Positive samples
                        "I love this product, totally worth it!",
                        "Amazing experience! Highly recommend!",
                        "Wonderful service, very satisfied!",
                        "Excellent quality, exceeded expectations!",
                        "Fantastic! Best purchase ever!",
                        "Great product, love it so much!",
                        "Perfect! Exactly what I wanted!",
                        "Awesome quality, very happy!",
                        "Delightful experience, will buy again!",
                        "Beautiful design and great functionality!",
                    ] * 100 + [
                        # Neutral samples
                        "It was okay, not too bad.",
                        "Average product, nothing special.",
                        "Decent quality for the price.",
                        "Fine, meets basic expectations.",
                        "Normal experience, as expected.",
                        "Moderate quality, acceptable.",
                        "Standard product, nothing extraordinary.",
                        "Okay service, could be better.",
                        "Acceptable quality, reasonable price.",
                        "Fair product, does the job.",
                    ] * 100 + [
                        # Negative samples
                        "Terrible experience, I'll never use it again.",
                        "Awful product, complete waste of money!",
                        "Horrible quality, very disappointed!",
                        "Worst purchase ever, broken on arrival!",
                        "Bad service, very frustrating!",
                        "Poor quality, doesn't work properly!",
                        "Hate it! Total disappointment!",
                        "Angry about this purchase, terrible!",
                        "Disappointing quality, not worth it!",
                        "Sad experience, wouldn't recommend!",
                    ] * 100,
                    'sentiment': ['Positive'] * 1000 + ['Neutral'] * 1000 + ['Negative'] * 1000
                }
                
                sample_df = pd.DataFrame(sample_data)
                sample_df.to_csv('sample_dataset.csv', index=False)
                st.success("✅ Sample dataset created: 'sample_dataset.csv'")
                st.info("You can now upload this file below to train the model!")
            
            st.markdown("---")
            
            # File uploaders for each dataset
            st.markdown("##### 📁 Upload Your Datasets")
            st.info("💡 Accepts both CSV and Excel (.xlsx, .xls) files")
            
            amazon_file = st.file_uploader("📦 Amazon Reviews (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key='amazon')
            sentiment140_file = st.file_uploader("🐦 Sentiment140 (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key='sentiment140')
            airline_file = st.file_uploader("✈️ Airline Sentiment (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key='airline')
            
            # Preview uploaded files
            if airline_file or amazon_file or sentiment140_file:
                st.markdown("---")
                st.markdown("##### 👀 Dataset Preview")
                
                if airline_file:
                    try:
                        # Check file type and read accordingly
                        if airline_file.name.endswith(('.xlsx', '.xls')):
                            preview_df = pd.read_excel(airline_file, nrows=5)
                        else:
                            preview_df = pd.read_csv(airline_file, nrows=5)
                        st.write("**Airline Dataset Preview:**")
                        st.write(f"Columns: {list(preview_df.columns)}")
                        st.dataframe(preview_df.head())
                        airline_file.seek(0)  # Reset file pointer
                    except Exception as e:
                        st.error(f"Error previewing airline file: {e}")
                
                if amazon_file:
                    try:
                        if amazon_file.name.endswith(('.xlsx', '.xls')):
                            preview_df = pd.read_excel(amazon_file, nrows=5)
                        else:
                            preview_df = pd.read_csv(amazon_file, nrows=5)
                        st.write("**Amazon Dataset Preview:**")
                        st.write(f"Columns: {list(preview_df.columns)}")
                        st.dataframe(preview_df.head())
                        amazon_file.seek(0)  # Reset file pointer
                    except Exception as e:
                        st.error(f"Error previewing amazon file: {e}")
                
                if sentiment140_file:
                    try:
                        if sentiment140_file.name.endswith(('.xlsx', '.xls')):
                            preview_df = pd.read_excel(sentiment140_file, nrows=5)
                        else:
                            preview_df = pd.read_csv(sentiment140_file, nrows=5, encoding='latin-1')
                        st.write("**Sentiment140 Dataset Preview:**")
                        st.write(f"Columns: {list(preview_df.columns)}")
                        st.dataframe(preview_df.head())
                        sentiment140_file.seek(0)  # Reset file pointer
                    except Exception as e:
                        st.error(f"Error previewing sentiment140 file: {e}")
            
            st.markdown("---")
            samples_per_class = st.slider("Samples per class", 1000, 10000, 5000, 500)
            
            if st.button("🚀 Train Model", key='train_btn'):
                dataframes = []
                
                # Load datasets
                with st.spinner("Loading datasets..."):
                    if amazon_file:
                        st.info("📦 Loading Amazon Reviews...")
                        df = DatasetLoader.load_amazon_reviews(amazon_file)
                        if df is not None:
                            dataframes.append(df)
                            st.success(f"✅ Loaded Amazon: {len(df)} samples")
                            st.info(f"Sentiments found: {df['sentiment'].unique()}")
                        else:
                            st.error("❌ Failed to load Amazon Reviews dataset")
                    
                    if sentiment140_file:
                        st.info("🐦 Loading Sentiment140...")
                        df = DatasetLoader.load_sentiment140(sentiment140_file)
                        if df is not None:
                            dataframes.append(df)
                            st.success(f"✅ Loaded Sentiment140: {len(df)} samples")
                            st.info(f"Sentiments found: {df['sentiment'].unique()}")
                        else:
                            st.error("❌ Failed to load Sentiment140 dataset")
                    
                    if airline_file:
                        st.info("✈️ Loading Airline Sentiment...")
                        df = DatasetLoader.load_airline_sentiment(airline_file)
                        if df is not None:
                            dataframes.append(df)
                            st.success(f"✅ Loaded Airline: {len(df)} samples")
                            st.info(f"Sentiments found: {df['sentiment'].unique()}")
                        else:
                            st.error("❌ Failed to load Airline Sentiment dataset")
                
                if not dataframes:
                    st.error("❌ No datasets were successfully loaded. Please check:")
                    st.error("1. File format is correct CSV")
                    st.error("2. Files are not corrupted")
                    st.error("3. Files contain the expected columns")
                    return
                
                if dataframes:
                    # Combine and balance
                    combined_df = DatasetLoader.combine_and_balance_datasets(
                        dataframes, samples_per_class
                    )
                    
                    if combined_df is None:
                        st.error("❌ Failed to create combined dataset. Please check your data.")
                        return
                    
                    # Check if we have valid data
                    if len(combined_df) < 100:
                        st.error("❌ Not enough data after processing. Please upload larger datasets.")
                        return
                    
                    # Save combined dataset
                    combined_df.to_csv('combined_sentiment_dataset.csv', index=False)
                    st.success(f"✅ Combined dataset saved: {len(combined_df)} samples")
                    
                    # Clear the cache before training to ensure fresh model
                    st.cache_resource.clear()
                    
                    # Train model
                    vectorizer, model, metrics = train_model(combined_df, preprocessor)
                    
                    if vectorizer is None or model is None:
                        st.error("❌ Model training failed. Please check the error messages above.")
                        return
                    
                    st.success("✅ Model trained successfully!")
                    st.balloons()
                    
                    # Force update the session state
                    st.session_state.model_trained = True
                    st.session_state.training_samples = len(combined_df)
                    
                    # Clear cache again to force reload
                    st.cache_resource.clear()
                    
                    # Wait a moment then rerun
                    import time
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Please upload at least one dataset")
        else:
            st.success("✅ Model loaded successfully!")
            
            if training_samples > 0:
                st.metric("Training Samples", f"{training_samples:,}")
            else:
                st.warning("Training samples: Unknown")
            
            if metrics:
                st.markdown("#### 📊 Model Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                with col2:
                    st.metric("F1 Score", f"{metrics.get('f1_score', 0):.2%}")
                
                # Show training date if available
                try:
                    with open('sentiment_model.pkl', 'rb') as f:
                        temp_data = pickle.load(f)
                        if 'training_date' in temp_data:
                            st.info(f"📅 Trained: {temp_data['training_date']}")
                        if 'classes' in temp_data:
                            st.info(f"🎯 Classes: {', '.join(temp_data['classes'])}")
                except:
                    pass
            
            st.markdown("---")
            
            if st.button("🔄 Retrain Model"):
                if os.path.exists('sentiment_model.pkl'):
                    os.remove('sentiment_model.pkl')
                if os.path.exists('combined_sentiment_dataset.csv'):
                    os.remove('combined_sentiment_dataset.csv')
                st.session_state.model_trained = False
                st.cache_resource.clear()  # Clear cached resources
                st.success("✅ Model deleted! Please upload datasets again.")
                st.rerun()
    
    # Main content
    st.markdown("<h1>✨ Social Media Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b21a8; font-size: 1.1rem; margin-bottom: 2rem;'>AI-powered emotion detection for tweets, reviews, and opinions</p>", unsafe_allow_html=True)
    
    # Check if model is available
    if vectorizer is None or model is None:
        st.warning("⚠️ Please upload datasets and train the model using the sidebar.")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Analyze Text", "📊 Batch Analysis", "📈 Model Insights", "☁️ Word Clouds", "📚 Dataset Info"])
    
    # Tab 1: Single Text Analysis
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💬 Enter Your Text")
            user_input = st.text_area(
                "",
                placeholder="Type or paste your social media text, tweet, comment, or review here...",
                height=150,
                label_visibility="collapsed"
            )
            
            analyze_btn = st.button("🔍 Analyze Sentiment", key="analyze_single")
            
            if analyze_btn and user_input.strip():
                with st.spinner("Analyzing..."):
                    sentiment, confidence = predict_sentiment(user_input, vectorizer, model, preprocessor)
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Result")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, {get_sentiment_color(sentiment)}20, {get_sentiment_color(sentiment)}40); 
                             border-radius: 15px; border: 2px solid {get_sentiment_color(sentiment)}'>
                            <div style='font-size: 4rem; margin-bottom: 0.5rem;'>{get_sentiment_emoji(sentiment)}</div>
                            <h2 style='color: {get_sentiment_color(sentiment)}; margin: 0;'>{sentiment}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col2:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea20, #764ba240); 
                             border-radius: 15px; border: 2px solid #667eea'>
                            <h3 style='color: #667eea; margin: 0;'>Confidence</h3>
                            <div style='font-size: 3rem; font-weight: bold; color: #667eea; margin-top: 0.5rem;'>{confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col3:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f093fb20, #f5576c40); 
                             border-radius: 15px; border: 2px solid #f093fb'>
                            <h3 style='color: #f093fb; margin: 0;'>Status</h3>
                            <div style='font-size: 1.5rem; font-weight: bold; color: #f093fb; margin-top: 1rem;'>✓ Analyzed</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📚 Sentiment Keywords")
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #a855f720, #a855f740); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                <h4 style='color: #a855f7; margin: 0 0 0.5rem 0;'>😊 Positive</h4>
                <p style='font-size: 0.85rem; color: #6b21a8;'>love, happy, awesome, amazing, great, wonderful, fantastic, excellent, perfect, joy, delightful, beautiful</p>
            </div>
            
            <div style='background: linear-gradient(135deg, #3b82f620, #3b82f640); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                <h4 style='color: #3b82f6; margin: 0 0 0.5rem 0;'>😐 Neutral</h4>
                <p style='font-size: 0.85rem; color: #1e40af;'>okay, average, fine, decent, normal, expected, moderate, acceptable, standard, unsure</p>
            </div>
            
            <div style='background: linear-gradient(135deg, #ec489920, #ec489940); padding: 1rem; border-radius: 10px;'>
                <h4 style='color: #ec4899; margin: 0 0 0.5rem 0;'>😠 Negative</h4>
                <p style='font-size: 0.85rem; color: #9f1239;'>bad, terrible, awful, hate, disappointed, worst, horrible, broken, angry, poor, frustrating, sad</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 2: Batch Analysis
    with tab2:
        st.markdown("### 📁 Upload CSV for Batch Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file with 'text' column",
            type=['csv'],
            help="CSV should contain a 'text' column with comments/tweets"
        )
        
        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)
            
            if 'text' in df_upload.columns:
                st.success(f"✅ Loaded {len(df_upload)} records")
                
                if st.button("🚀 Analyze All", key="batch_analyze"):
                    with st.spinner("Analyzing batch..."):
                        predictions = []
                        confidences = []
                        
                        progress = st.progress(0)
                        for idx, text in enumerate(df_upload['text']):
                            try:
                                pred, conf = predict_sentiment(str(text), vectorizer, model, preprocessor)
                                predictions.append(pred)
                                confidences.append(conf)
                            except:
                                predictions.append('Unknown')
                                confidences.append(0)
                            
                            progress.progress((idx + 1) / len(df_upload))
                        
                        df_upload['predicted_sentiment'] = predictions
                        df_upload['confidence'] = confidences
                        
                        st.markdown("### 📊 Results")
                        st.dataframe(df_upload, use_container_width=True)
                        
                        csv = df_upload.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "sentiment_results.csv",
                            "text/csv"
                        )
                        
                        sentiment_counts = df_upload['predicted_sentiment'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color=sentiment_counts.index,
                            color_discrete_map={'Positive': '#a855f7', 'Neutral': '#3b82f6', 'Negative': '#ec4899'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ CSV must contain a 'text' column")
    
    # Tab 3: Model Insights
    with tab3:
        st.markdown("### 📊 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Accuracy</h3>
                <p style='font-size: 1.8rem; margin: 0;'>{metrics.get('accuracy', 0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Precision</h3>
                <p style='font-size: 1.8rem; margin: 0;'>{metrics.get('precision', 0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Recall</h3>
                <p style='font-size: 1.8rem; margin: 0;'>{metrics.get('recall', 0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>F1 Score</h3>
                <p style='font-size: 1.8rem; margin: 0;'>{metrics.get('f1_score', 0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            st.markdown("### 🎯 Confusion Matrix")
            
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                       xticklabels=['Positive', 'Neutral', 'Negative'],
                       yticklabels=['Positive', 'Neutral', 'Negative'],
                       ax=ax)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
            st.pyplot(fig)
        
        # Classification Report
        if 'classification_report' in metrics:
            st.markdown("### 📋 Detailed Classification Report")
            st.text(metrics['classification_report'])
    
    # Tab 4: Word Clouds
    with tab4:
        st.markdown("### ☁️ Sentiment Word Clouds")
        
        if os.path.exists('combined_sentiment_dataset.csv'):
            df_dataset = pd.read_csv('combined_sentiment_dataset.csv')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 😊 Positive Words")
                pos_texts = df_dataset[df_dataset['sentiment'] == 'Positive']['text'].head(500).tolist()
                if pos_texts:
                    cleaned_pos = [preprocessor.clean_text(t) for t in pos_texts]
                    fig_pos = create_wordcloud(cleaned_pos, 'Positive')
                    if fig_pos:
                        st.pyplot(fig_pos)
            
            with col2:
                st.markdown("#### 😠 Negative Words")
                neg_texts = df_dataset[df_dataset['sentiment'] == 'Negative']['text'].head(500).tolist()
                if neg_texts:
                    cleaned_neg = [preprocessor.clean_text(t) for t in neg_texts]
                    fig_neg = create_wordcloud(cleaned_neg, 'Negative')
                    if fig_neg:
                        st.pyplot(fig_neg)
            
            st.markdown("#### 😐 Neutral Words")
            neu_texts = df_dataset[df_dataset['sentiment'] == 'Neutral']['text'].head(500).tolist()
            if neu_texts:
                cleaned_neu = [preprocessor.clean_text(t) for t in neu_texts]
                fig_neu = create_wordcloud(cleaned_neu, 'Neutral')
                if fig_neu:
                    st.pyplot(fig_neu)
        else:
            st.info("Word clouds will be generated after training the model with real datasets.")
    
    # Tab 5: Dataset Info
    with tab5:
        st.markdown("### 📚 Dataset Information")
        
        if os.path.exists('combined_sentiment_dataset.csv'):
            df_info = pd.read_csv('combined_sentiment_dataset.csv')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Dataset Statistics")
                st.metric("Total Samples", f"{len(df_info):,}")
                st.metric("Unique Texts", f"{df_info['text'].nunique():,}")
                
                # Sentiment distribution
                sentiment_dist = df_info['sentiment'].value_counts()
                st.markdown("##### Sentiment Distribution")
                for sent, count in sentiment_dist.items():
                    percentage = (count / len(df_info)) * 100
                    st.metric(f"{get_sentiment_emoji(sent)} {sent}", f"{count:,} ({percentage:.1f}%)")
            
            with col2:
                st.markdown("#### 📈 Distribution Chart")
                sentiment_counts = df_info['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                fig = px.bar(
                    sentiment_counts,
                    x='Sentiment',
                    y='Count',
                    color='Sentiment',
                    color_discrete_map={'Positive': '#a855f7', 'Neutral': '#3b82f6', 'Negative': '#ec4899'},
                    title="Sentiment Class Balance"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 📄 Sample Data")
            
            # Show samples from each category
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                with st.expander(f"{get_sentiment_emoji(sentiment)} {sentiment} Samples"):
                    samples = df_info[df_info['sentiment'] == sentiment].sample(min(5, len(df_info[df_info['sentiment'] == sentiment])))
                    for idx, row in samples.iterrows():
                        st.markdown(f"- {row['text'][:200]}...")
        else:
            st.info("📦 Dataset information will be available after uploading and processing the data.")
            
            st.markdown("---")
            st.markdown("### 📖 How to Use This App")
            
            st.markdown("""
            #### Step 1: Upload Datasets
            - Use the **sidebar** to upload one or more of the following datasets:
              - **Amazon Product Reviews** - Product reviews with ratings
              - **Sentiment140** - Twitter sentiment dataset (1.6M tweets)
              - **Twitter US Airline Sentiment** - Airline customer tweets
            
            #### Step 2: Train the Model
            - Select the number of samples per class (1000-10000)
            - Click **"🚀 Train Model"**
            - The app will:
              - Load and preprocess all datasets
              - Combine and balance the data
              - Train a Logistic Regression model with TF-IDF
              - Save the model for future use
            
            #### Step 3: Analyze Sentiments
            - Go to **"🎯 Analyze Text"** tab to analyze individual texts
            - Use **"📊 Batch Analysis"** for multiple texts at once
            - View **"📈 Model Insights"** for performance metrics
            - Explore **"☁️ Word Clouds"** for visual insights
            
            #### Expected Dataset Formats:
            
            **Amazon Reviews:**
            - Columns: `review_text` or `review`, `rating` (1-5)
            - Or: `text`, `sentiment`
            
            **Sentiment140:**
            - Columns: target (0/2/4), ids, date, flag, user, text
            - Format: CSV without headers
            
            **Airline Sentiment:**
            - Columns: `text`, `airline_sentiment` (positive/neutral/negative)
            """)
            
            st.markdown("---")
            st.markdown("### 🔗 Dataset Sources")
            
            st.markdown("""
            - 📦 [Amazon Product Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
            - 🐦 [Sentiment140 (Twitter)](https://www.kaggle.com/datasets/kazanova/sentiment140)
            - ✈️ [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        <p>Developed with 💜 by <strong>amitt</strong> | College ML Project 2025</p>
        <p style='font-size: 0.9rem; color: #9333ea;'>Machine Learning • Natural Language Processing • Sentiment Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()