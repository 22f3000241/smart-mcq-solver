# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')


#pip install wandb


import wandb

wandb.init(
    project="22f3000241-t22026"
    
)



import matplotlib.pyplot as plt



train_df = pd.read_csv('/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv')
sample_sub = pd.read_csv('/kaggle/input/competitions/smart-mcq-solver-challenge/sample_submission.csv')


train_df.head()


print(train_df.shape)
print(test_df.shape)


print(f'train columns: {train_df.columns}')
print(f'test columns: {test_df.columns}')


train_df.isnull().sum()


test_df.isnull().sum()





train_df['answer'].describe()


answer_count = train_df['answer'].value_counts()
answer_count


percentages = (train_df['answer'].value_counts()/len(train_df))*100
percentages



ax = answer_count.plot(kind='bar')
for i, p in enumerate(percentages):
    ax.text(i, answer_count.iloc[i] + 5, f'{p:.1f}%', ha='center')
plt.xlabel('Answer')
plt.ylabel('Count')
plt.title('Distribution of Answers')
plt.show()





train_df['prompt'].str.len()


train_df['prompt'].describe()


print(f'Total number of duplicates present in prompt : {train_df['prompt'].duplicated().sum()}')


print(f'Total number of unique prompts : {train_df['prompt'].nunique()}')


train_df[train_df['prompt'].duplicated()].head()





len(train_df)


train_df['id']


options=['A','B','C','D']


options=['A','B','C','D']

avg_lengths = []

for option in options:
    avg_lengths.append(train_df[option].str.len().mean())

print(avg_lengths)
    



plt.bar(options, avg_lengths)
plt.xlabel("Option")
plt.ylabel("Average Length")
plt.title("Average Length of Each Option")
plt.show()


train_df['correct_len'] = train_df.apply(
    lambda row: len(row[row['answer']]),
    axis=1
)
correct_len = train_df['correct_len'].mean()
correct_len
print(f'Average of correct answer length: {correct_len:.2f}')


avg_correct = train_df['correct_len'].mean()

plt.bar(options + ['Correct'], avg_lengths + [avg_correct])
plt.xlabel('Option')
plt.ylabel('Average Length')
plt.title('Average Length of Options vs Correct Answers')
plt.show()





train_prompt=set(train_df['prompt'])



test_df['in_train'] = test_df['prompt'].isin(train_prompt)
print(f'No of questions present in train dataset which is also present in the test dataset : {test_df['in_train'].sum()} out of {len(test_df)}') 
print(f'In Percentage : {test_df['in_train'].mean()*100}%')




lookup = dict(zip(train_df['prompt'], train_df['answer']))
test_df['known_answer'] = test_df['prompt'].map(lookup)



print(test_df[['id', 'prompt', 'known_answer']].dropna())





test_df.columns


test_df.head()





def make_pred(known_answer):
    freq_order = ['B', 'C', 'A', 'D', 'E']
    
    # Step 2: remove known_answer from the list
    remaining = [x for x in freq_order if x != known_answer ]
    
    # Step 3: take first 2
    top2 = remaining[:2]
    
    # Step 4: combine known_answer + top2, join with space
    pred = known_answer + " " + top2[0] + " " + top2[1]
    
    return pred


predictions = []
for i, row in test_df.iterrows():
    if pd.notna(row['known_answer']):
        pred = make_pred(row['known_answer'])
    else:
        pred = "B C A"
    predictions.append(pred)


len(predictions)


sub=pd.read_csv('/kaggle/input/competitions/smart-mcq-solver-challenge/sample_submission.csv')
sub


submission = pd.DataFrame({
    "ID":test_df['id'],
    "Prediction":predictions
})

submission.to_csv("submission.csv",index=False)

print(submission.head())


get_ipython().getoutput("git config --global user.name "Ashish Augustine"")
get_ipython().getoutput("git config --global user.email "22f3000241@ds.study.iitm.ac.in"")

get_ipython().getoutput("git clone https://github.com/22f3000241/smart-mcq-solver.git")


get_ipython().run_line_magic("cd", " smart-mcq-solver")


get_ipython().getoutput("pwd")
get_ipython().getoutput("ls")


get_ipython().getoutput("cp /kaggle/working/.virtual_documents/__notebook_source__.ipynb .")


get_ipython().getoutput("ls")


get_ipython().getoutput("git status")


get_ipython().getoutput("git add __notebook_source__.ipynb")


get_ipython().getoutput("git status")


get_ipython().getoutput("git commit -m "Add Kaggle notebook"")


get_ipython().getoutput("git config --global user.name "Ashish Augustine"")
get_ipython().getoutput("git config --global user.email "22f3000241@ds.study.iitm.ac.in"")


get_ipython().getoutput("git push origin main")
