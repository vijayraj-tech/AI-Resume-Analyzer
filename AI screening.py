# AI Resume Screening System (Final Working Version)

import PyPDF2
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore

# ==============================
# 📂 INPUT (Dynamic Resume)
# ==============================

filename = input("Enter resume file name (e.g. Data Scientist-1.pdf): ")
file_path = f'Datasets (Resume:CV)/{filename}'

try:
    file = open(file_path, 'rb')
except FileNotFoundError:
    print(Fore.RED + "❌ File not found! Check file name and path.")
    exit()

reader = PyPDF2.PdfReader(file)

number_of_pages = len(reader.pages)
print(f"\n📄 Resume contains {number_of_pages} pages")

# ==============================
# 📄 TEXT EXTRACTION
# ==============================

content = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        content += text

file.close()

# ==============================
# 🧹 TEXT CLEANING
# ==============================

content = content.lower()
content = re.sub(r'[0-9]+', '', content)
content = content.translate(str.maketrans('', '', string.punctuation))

# ==============================
# 🧠 KEYWORD DICTIONARY
# ==============================

Area_with_key_term = {
    'Data science': ['algorithm', 'analytics', 'machine learning', 'data mining', 'python', 'statistics'],
    'Programming': ['python', 'java', 'c++', 'javascript', 'sql'],
    'Experience': ['project', 'company', 'experience', 'internship'],
    'Data analytics': ['data analysis', 'pandas', 'numpy', 'visualization'],
    'Machine learning': ['supervised', 'unsupervised', 'nlp', 'deep learning'],
    'Software': ['django', 'react', 'node.js', 'html', 'css'],
    'Web skill': ['web design', 'seo', 'marketing'],
    'Personal Skill': ['leadership', 'teamwork', 'communication'],
    'Language': ['english', 'hindi']
}

# ==============================
# 📊 SCORING
# ==============================

scores = {}

for domain in Area_with_key_term:
    score = 0
    for word in Area_with_key_term[domain]:
        if word in content:
            score += 1
    scores[domain] = score

# ==============================
# 📈 RESULT DISPLAY
# ==============================

df = pd.DataFrame(scores.items(), columns=['Domain', 'Score'])
df = df.sort_values(by='Score', ascending=False)

print("\n📊 Resume Analysis Result:\n")
print(df)

total_score = sum(scores.values())
print(Fore.BLUE + f"\n🔢 Total Score: {total_score}")

# ==============================
# 🎯 DECISION LOGIC
# ==============================

if total_score >= 10:
    print(Fore.GREEN + "✅ Status: Strong Profile")
elif total_score >= 5:
    print(Fore.YELLOW + "⚠️ Status: Average Profile")
else:
    print(Fore.RED + "❌ Status: Weak Profile")

# ==============================
# 📊 VISUALIZATION
# ==============================

plt.figure(figsize=(8,6))
plt.barh(df['Domain'], df['Score'])
plt.xlabel("Score")
plt.title("Resume Skill Analysis")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()