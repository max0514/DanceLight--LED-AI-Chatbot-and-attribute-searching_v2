import pandas as pd

df = pd.read_csv('report_docling.csv')

total = len(df)
passed = len(df[df['Result'] == '✅ Pass'])
failed = total - passed
accuracy = (passed / total) * 100
avg_time = df['Avg Time (s)'].mean()
avg_consistency = df['Consistency %'].mean()

print(f"## 📊 Model Testing Final Report")
print(f"- **Total Cases Tested:** {total}")
print(f"- **Passed:** {passed}")
print(f"- **Failed:** {failed}")
print(f"- **Overall Accuracy:** {accuracy:.1f}%")
print(f"- **Average Response Time:** {avg_time:.2f} seconds per query")
print(f"- **Average Consistency Score:** {avg_consistency:.1f}%\n")

if failed > 0:
    print(f"### ❌ Breakdown of Failed Cases")
    failed_df = df[df['Result'] == '❌ Fail']
    for idx, row in failed_df.iterrows():
        print(f"**Test ID {row['ID']}**")
        print(f"- **User Query:** {row['Query']}")
        print(f"- **Expected Model Mentioned:** {row['Expected']}")
        
        # Get just the first 1-2 lines of the AI answer to keep it readable
        ai_resp = str(row['AI Answer']).split('\n')
        short_resp = ai_resp[0]
        if len(ai_resp) > 1 and len(short_resp) < 50:
            short_resp += " " + ai_resp[1]
        
        print(f"- **AI Top Answer Prefix:** {short_resp} ...\n")
