import pandas as pd
import numpy as np

import missingno as msno


DROP_FEATURES = [
  'source',
  'city',
  'lastName',
  'firstName',
  'selfMade',
  'title',
  'rank',
  'state',
  'residenceStateRegion',
  'date',
  'personName',
  'birthDay',
  'birthDate',
  'birthYear',
  'organization'
]

def get_X(df:pd.DataFrame, drop_features:list=DROP_FEATURES):
  '''Make feature vectors from a DataFrame.

  Args:
      df: DataFrame
      features: selected columns
  '''
  df = df.copy()
  df['gdp_country'] = df['gdp_country'].str.replace('$','')
  df['gdp_country'] = df['gdp_country'].str.replace(',','')
  df['gdp_country'] = df['gdp_country'].astype('float64')

  # 나이 빈 데이터 평균값으로 대체
  df['age'].fillna(df['age'].mean(), inplace=True)
  
  # country 빈 데이터 시민권으로 대체
  df['country'].fillna(df['countryOfCitizenship'],inplace=True)
  
  # 시민권별로 그룹화하고 윗 내용으로 결측값 처리
  df = df.groupby('countryOfCitizenship').fillna(method='ffill') 
  df['city'].fillna('New York',inplace=True)
  df.fillna(df.mean(numeric_only=True),inplace=True)
  
  '''
  # 원핫인 코딩을 위한 처리
  df['gender'] = df['gender'].map({'F': 0, 'M': 1})
  df['status'] = df['status'].map({'U': 0, 'D': 1,'N': 2, 'Split Family Fortune': 3,'E': 4, 'R': 5})
  df['category'] = df['category'].map({'Fashion & Retail': 0, 'Automotive': 1,'Technology': 2, 'Finance & Investments': 3,'Media & Entertainment': 4, 'Telecom': 5,'Diversified': 6, 'Food & Beverage': 7,'Logistics': 8, 'Gambling & Casinos': 9, 'Manufacturing': 10,'Real Estate': 11, 'Metals & Mining': 12, 'Energy': 13,'Healthcare': 14, 'Service': 15,'Construction & Engineering': 16, 'Sports': 17})
  '''
      
  # 나이 10대 별로 분배
  def category_age(x):
        if x < 20:
            return 0
        elif x < 30:
            return 1
        elif x < 40:
            return 2
        elif x < 50:
            return 3
        elif x < 60:
            return 4
        elif x < 70:
            return 5
        elif x < 80:
            return 6
        elif x < 90:
            return 7
        elif x < 100:
            return 8
        else:
            return 9

  df['age'] = df['age'].apply(category_age)
  
  def category_month(x):
      if x < 3:
          return 0
      elif x < 6:
          return 1
      elif x < 9:
          return 2
      elif x < 12:
          return 3
      else:
          return 0
      
  df['birthMonth'] = df['birthMonth'].apply(category_month)
  # 필요없는 columns delete
  # .dropna(thresh=int(len(data)*0.5),axis=1) 50% 이상 데이터가 없을 경우 삭제
  df.drop(drop_features, axis=1, inplace=True)
  '''
  # 원-핫 인코딩
  pd.get_dummies(df['status']).to_numpy(dtype=np.float64)
  pd.get_dummies(df['gender']).to_numpy(dtype=np.float64)
  pd.get_dummies(df['category']).to_numpy(dtype=np.float64)
  pd.get_dummies(df['age']).to_numpy(dtype=np.float64)
  pd.get_dummies(df['birthMonth']).to_numpy(dtype=np.float64)
  '''
  print('Missing Value:\n',df.isnull().sum())
  # msno.matrix(df)
  return pd.get_dummies(df)

def get_y(df:pd.DataFrame):
  '''Make the target from a DataFrame.

  Args:
      df: DataFrame
  '''
  print(df['selfMade'].value_counts())
  return df.selfMade.to_numpy(dtype=np.float64)