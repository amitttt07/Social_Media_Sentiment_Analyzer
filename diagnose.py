"""
Diagnostic Script to Check Your Dataset File
Run this separately to see exactly what's in your file
"""

import pandas as pd

# CHANGE THIS to your file path
FILE_PATH = "Tweets.csv"  # or "Tweets.xlsx"

print("=" * 60)
print("DATASET DIAGNOSTIC REPORT")
print("=" * 60)

try:
    # Try to load the file
    print(f"\n📂 Loading file: {FILE_PATH}")
    
    if FILE_PATH.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
        print("✅ Loaded as Excel file")
    else:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(FILE_PATH, encoding=encoding)
                print(f"✅ Loaded as CSV with {encoding} encoding")
                break
            except:
                continue
    
    print(f"✅ File loaded successfully!")
    print(f"📊 Total rows: {len(df)}")
    print(f"📊 Total columns: {len(df.columns)}")
    
    # Show columns
    print(f"\n📋 All Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Check for sentiment columns
    print(f"\n🔍 Looking for sentiment columns...")
    sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
    
    if sentiment_cols:
        print(f"✅ Found sentiment columns: {sentiment_cols}")
        
        for col in sentiment_cols:
            print(f"\n📊 Analysis of column: '{col}'")
            print(f"  Data type: {df[col].dtype}")
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Null values: {df[col].isnull().sum()}")
            
            print(f"\n  📈 Value Distribution:")
            value_counts = df[col].value_counts()
            for value, count in value_counts.items():
                percentage = (count / len(df)) * 100
                print(f"    '{value}': {count} ({percentage:.1f}%)")
            
            print(f"\n  🎯 All unique values:")
            print(f"    {df[col].unique()}")
            
            # Show first 10 values
            print(f"\n  📝 First 10 values:")
            for i, val in enumerate(df[col].head(10), 1):
                print(f"    {i}. '{val}'")
    else:
        print("❌ No sentiment column found!")
        print("   Looking for any column with 'sentiment' in name")
    
    # Check for text column
    print(f"\n🔍 Looking for text columns...")
    text_cols = [col for col in df.columns if 'text' in col.lower()]
    
    if text_cols:
        print(f"✅ Found text columns: {text_cols}")
        for col in text_cols:
            print(f"\n  Sample texts from '{col}':")
            for i, text in enumerate(df[col].head(5), 1):
                print(f"    {i}. {str(text)[:100]}...")
    else:
        print("❌ No text column found!")
    
    # Show first 5 rows
    print(f"\n📄 First 5 Rows Preview:")
    print(df.head())
    
    # Check for specific Twitter Airline columns
    expected_cols = ['airline_sentiment', 'text']
    print(f"\n✅ Checking for expected columns: {expected_cols}")
    for col in expected_cols:
        if col in df.columns:
            print(f"  ✅ '{col}' - FOUND")
        else:
            print(f"  ❌ '{col}' - NOT FOUND")
    
    # Test the transformation
    if 'airline_sentiment' in df.columns:
        print(f"\n🧪 Testing transformation...")
        test_df = df[['text', 'airline_sentiment']].copy()
        test_df = test_df.rename(columns={'airline_sentiment': 'sentiment'})
        
        print(f"Before capitalization:")
        print(test_df['sentiment'].value_counts())
        
        test_df['sentiment'] = test_df['sentiment'].str.strip().str.capitalize()
        
        print(f"\nAfter capitalization:")
        print(test_df['sentiment'].value_counts())
        
        print(f"\nUnique sentiments after processing: {test_df['sentiment'].unique()}")
        
        if test_df['sentiment'].nunique() >= 2:
            print("✅ SUCCESS! File has multiple sentiment classes!")
        else:
            print("❌ PROBLEM! File only has one sentiment class after processing!")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    print("\nFull error:")
    print(traceback.format_exc())

print("\n💡 Next Steps:")
print("1. Check if the sentiment column exists")
print("2. Verify there are multiple unique sentiment values")
print("3. Make sure 'text' column exists with actual text data")
print("4. If everything looks good, the Streamlit app should work!")