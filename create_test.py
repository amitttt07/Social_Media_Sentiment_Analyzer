"""
Create a Test Dataset File
This creates a properly formatted file to test the app
"""

import pandas as pd

# Create sample data with all three sentiments
data = {
    'text': [
        # Positive tweets
        "@VirginAmerica This is amazing! Best flight ever!",
        "@VirginAmerica Thanks for the excellent service!",
        "@VirginAmerica Great experience, highly recommend!",
        "@VirginAmerica Perfect flight, wonderful staff!",
        "@VirginAmerica Best airline I've ever flown with!",
        "@VirginAmerica Amazing journey, thank you so much!",
        "@VirginAmerica Fantastic service and comfortable seats!",
        "@VirginAmerica Love this airline, will fly again!",
        "@VirginAmerica Exceeded all my expectations!",
        "@VirginAmerica Brilliant experience from start to finish!",
    ] * 100 + [
        # Neutral tweets
        "@United Flight was okay, nothing special.",
        "@United Average experience, as expected.",
        "@United It was fine, got from A to B.",
        "@United Normal flight, no complaints.",
        "@United Standard service, adequate.",
        "@United Decent flight, nothing memorable.",
        "@United Acceptable journey, reasonable.",
        "@United Ordinary experience, as usual.",
        "@United Fair service, meets expectations.",
        "@United Regular flight, nothing outstanding.",
    ] * 100 + [
        # Negative tweets
        "@SouthwestAir This is terrible! Worst experience!",
        "@SouthwestAir Very disappointed with the service!",
        "@SouthwestAir Horrible flight, never again!",
        "@SouthwestAir Awful experience, wasted money!",
        "@SouthwestAir Terrible customer service, angry!",
        "@SouthwestAir Worst airline ever, frustrated!",
        "@SouthwestAir Disappointing journey, not happy!",
        "@SouthwestAir Bad experience, will not recommend!",
        "@SouthwestAir Poor service, very upset!",
        "@SouthwestAir Unacceptable treatment, disgusted!",
    ] * 100,
    
    'sentiment': (
        ['Positive'] * 1000 + 
        ['Neutral'] * 1000 + 
        ['Negative'] * 1000
    )
}

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("=" * 60)
print("CREATING TEST DATASET")
print("=" * 60)

print(f"\n📊 Dataset Info:")
print(f"  Total rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")

print(f"\n📈 Sentiment Distribution:")
print(df['sentiment'].value_counts())

print(f"\n🎯 Unique sentiments: {df['sentiment'].unique()}")

# Save as CSV
csv_file = "test_airline_sentiment.csv"
df.to_csv(csv_file, index=False)
print(f"\n✅ Saved as CSV: {csv_file}")

# Save as Excel
excel_file = "test_airline_sentiment.xlsx"
df.to_excel(excel_file, index=False, engine='openpyxl')
print(f"✅ Saved as Excel: {excel_file}")

print("\n📄 First 5 rows:")
print(df.head())

print("\n📄 Last 5 rows:")
print(df.tail())

print("\n" + "=" * 60)
print("TEST FILES CREATED SUCCESSFULLY!")
print("=" * 60)

print("\n💡 Next Steps:")
print(f"1. Upload '{csv_file}' OR '{excel_file}' to the Streamlit app")
print("2. These files are guaranteed to work!")
print("3. Use these to test if the app logic is correct")
print("4. If these work, compare with your Kaggle file to see the difference")

print("\n🔍 To verify your Kaggle file:")
print("  Run the diagnostic script on your actual Tweets.csv/xlsx file")
print("  Compare the output with this test file")