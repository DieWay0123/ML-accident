from matplotlib.hatch import HorizontalHatch
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import wget
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

def read_csv_file():
  # read (我將放在github上並讀取 因為放在colab好像會消失)
  file_path = "https://raw.githubusercontent.com/daniel88516/112---A1-A2-/main/112%E5%B9%B4-%E8%87%BA%E5%8C%97%E5%B8%82A1%E5%8F%8AA2%E9%A1%9E%E4%BA%A4%E9%80%9A%E4%BA%8B%E6%95%85%E6%98%8E%E7%B4%B0.csv"
  ref_file_path = "https://github.com/Killer66562/accident/raw/refs/heads/main/%E4%BA%A4%E9%80%9A%E4%BA%8B%E6%95%85%E4%BB%A3%E7%A2%BC%E5%B0%8D%E7%85%A7%E8%A1%A8.csv"

  data = pd.read_csv(file_path, encoding='big5', low_memory=False, dtype=str)
  ref = pd.read_csv(ref_file_path, encoding='big5', low_memory=False, dtype=str)

  return data, ref

def font():
  #wget.download("https://github.com/GrandmaCan/ML/raw/main/Resgression/ChineseFont.ttf")
  fm.fontManager.addfont("ChineseFont.ttf")
  plt.rc('font', family="ChineseFont")

def hourly_data():
  
  unique_case_data.loc[:, '發生時-Hours'] = pd.to_numeric(data.loc[:, '發生時-Hours'], errors='coerce')
  #print(unique_case_data)
  case_perHour_data = unique_case_data['發生時-Hours'].value_counts()
  
  plt.figure(figsize=(10, 6))
  plt.bar(case_perHour_data.index, case_perHour_data.values, color='skyblue', edgecolor='black')
  plt.xlabel('時間')
  plt.ylabel('事件數量')
  plt.xticks(range(0, 24))  
  plt.title('112年個時段事件發生總數分布圖')
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.show() 

if __name__ == '__main__':
  font()
  data, ref = read_csv_file()
  unique_case_data = data.drop_duplicates(subset=['發生月', '發生日', '發生時-Hours', '發生分', '肇事地點'])
  
  #行人傷亡數量(使用無去重資料)
  data_pedestrian = data[data['車種'] == 'H01']
  data_pedestrian = data_pedestrian.astype({'發生月': 'int32', '死亡人數': 'int64', '2-30日死亡人數': 'int64', '受傷人數': 'int64'})
  data_pedestrian_count = data_pedestrian.groupby(by=['發生月'])[['受傷人數', '2-30日死亡人數', '死亡人數']].sum()
  data_pedestrian_count['死亡人數'] = data_pedestrian_count['死亡人數'].add(data_pedestrian_count['2-30日死亡人數'])
  data_pedestrian_count = data_pedestrian_count.drop('2-30日死亡人數', axis=1)
  total_pedestrian_count = data_pedestrian_count.sum()
  
  #繪製行人受傷/死亡人數折線圖
  plt.figure(figsize=(12, 6))
  plt.plot(range(1, 13), data_pedestrian_count['受傷人數'], marker='o', linestyle='-', label='行人受傷人數')
  #折線圖每個標點加入數字
  i = 1
  for data in data_pedestrian_count['受傷人數']:
    plt.annotate(f'{data}', (i, data), textcoords="offset points", xytext=(0, 10), ha='center')
    i += 1
  
  plt.plot(range(1, 13), data_pedestrian_count['死亡人數'], marker='o', linestyle='dashed', label='行人死亡人數')
  i = 1
  for data in data_pedestrian_count['死亡人數']:
    plt.annotate(f'{data}', (i, data), textcoords="offset points", xytext=(0, 10), ha='center')
    i += 1
  
  #設定圖片標示及標題
  plt.legend()
  plt.xlabel('月份')
  plt.ylabel('人數')
  plt.title('112年度A1/A2事件行人受傷及死亡數折線圖')
  plt.text(11, 268, f'總受傷人數:{total_pedestrian_count["受傷人數"]}')
  plt.text(11, 258, f'總死亡人數:{total_pedestrian_count["死亡人數"]}')
  
  #調整圖片座標軸參數
  plt.ylim(0, 320)
  plt.xticks(range(1, 13))
  plt.show()
  
  age_intervals = {
    '0-12歲': list(range(0, 13)),
    '13-17歲': list(range(13, 18)),
    '18-24歲': list(range(18, 25)),
    '25-64歲': list(range(25, 65)),
    '65歲以上': list(range(65, 112))
  }
  replace_dict = {}
  for label, ages in age_intervals.items():
    for age in ages:
      replace_dict[str(age)] = label
      
  #age mapping
  data_pedestrian["年齡"] = data_pedestrian["年齡"].dropna().astype(str) \
    .replace(replace_dict)
  data_pedestrian_count_by_age = data_pedestrian.groupby(by=['年齡'])[['受傷人數', '2-30日死亡人數', '死亡人數']].sum()
  data_pedestrian_count_by_age['死亡人數'] = data_pedestrian_count_by_age['死亡人數'].add(data_pedestrian_count_by_age['2-30日死亡人數'])
  data_pedestrian_count_by_age = data_pedestrian_count_by_age.drop('2-30日死亡人數', axis=1)
  print(data_pedestrian_count_by_age)
  
  explode=(0.1, 0, 0, 0, 0)
  fig = plt.figure(figsize=(10, 6))
  data_sort_by_hurt = data_pedestrian_count_by_age.sort_values(by='受傷人數', ascending=False)
  ax1 = fig.add_subplot(121)
  ax1.pie(data_sort_by_hurt['受傷人數'], labels=data_sort_by_hurt.index, autopct='%1.1f%%', startangle=90, labeldistance=1.05, explode=explode)
  ax1.set_title('7到9點時段的事件區序分佈')
  ax1.set_ylabel('')
  ax1.axis('equal')
  
  data_sort_by_death = data_pedestrian_count_by_age.sort_values(by='死亡人數', ascending=False)
  data_sort_by_death = data_sort_by_death.loc[(data_sort_by_death['死亡人數']!=0)]
  ax2 = fig.add_subplot(122)
  ax2.pie(data_sort_by_death['死亡人數'], labels=data_sort_by_death.index, autopct='%1.1f%%', startangle=90, labeldistance=1.05)
  ax2.set_title('17到19點時段的事件區序分佈')
  ax2.set_ylabel('')
  ax2.axis('equal')
  plt.show()
  

  '''
  data.loc[:, '發生年度'] = data['發生年度'].replace("112", "2023")
  data['發生日期'] = data['發生年度'].str.cat(data['發生月'], sep='-')
  data['發生日期'] = pd.to_datetime(data['發生日期'].str.cat(data['發生日'], sep='-'))
  print(data.head)
  '''
  '''
  unique_case_data.loc[:, '發生時-Hours'] = pd.to_numeric(unique_case_data['發生時-Hours'], errors='coerce')
  area_morning_rushHour_data = unique_case_data[(unique_case_data['發生時-Hours'] >= 7) & (unique_case_data['發生時-Hours'] <= 9)]
  area_count_morning =  area_morning_rushHour_data['區序'].value_counts()
  
  area_night_rushHour_data = unique_case_data[(unique_case_data['發生時-Hours'] >= 17) & (unique_case_data['發生時-Hours'] <= 19)]
  area_count_night =  area_night_rushHour_data['區序'].value_counts()
  
  area_count_total = area_count_morning.add(area_count_night)

  area_label_morning = [s[2:] for s in area_count_morning.index.values]
  area_label_night = [s[2:] for s in area_count_night.index.values]
  
  area_count_morning = area_count_morning.sort_index()
  area_count_night = area_count_night.sort_index()
  area_count_total = area_count_total.sort_index()
  area_label_total = [s[2:] for s in area_count_total.index.values]
  
  explode=(0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  fig = plt.figure(figsize=(10, 6))
  ax1 = fig.add_subplot(121)
  ax1.pie(area_count_morning.sort_values(ascending=False), labels=area_label_morning, autopct='%1.1f%%', startangle=90, labeldistance=1.05, explode=explode)
  ax1.set_title('7到9點時段的事件區序分佈')
  ax1.set_ylabel('')
  ax1.axis('equal')
  
  ax2 = fig.add_subplot(122)
  ax2.pie(area_count_night.sort_values(ascending=False), labels=area_label_night, autopct='%1.1f%%', startangle=90, labeldistance=1.05, explode=explode)
  ax2.set_title('17到19點時段的事件區序分佈')
  ax2.set_ylabel('')
  ax2.axis('equal')

  plt.figure()
  plt.title("上午/晚間尖峰時段各區域交通事件總數統計")
  plt.bar(area_label_total, area_count_morning, label='早上尖峰時段(7~9)')
  plt.bar(area_label_total, area_count_night, bottom=area_count_morning, label='晚上尖峰時段(17~19)')
    
  for i in range(len(area_count_night)):
    plt.text(i, area_count_morning.iloc[i]/2-0.5, area_count_morning.iloc[i], fontsize=14, horizontalalignment='center')
    plt.text(i, area_count_night.iloc[i]/2 + area_count_morning.iloc[i] -0.5, area_count_night.iloc[i], fontsize=14, horizontalalignment='center')
    plt.text(i, area_count_total.iloc[i] + 1.5, area_count_total.iloc[i], fontsize=14, horizontalalignment='center')
  plt.legend()
  
  plt.show()
  
  zhongshan_area_data = unique_case_data[unique_case_data['區序'] == '03中山區']
  
  scatter_data = zhongshan_area_data[['發生時-Hours', '座標-X', '座標-Y']].dropna()  # 移除空值
  scatter_data['發生時-Hours'] = scatter_data['發生時-Hours'].astype(int)  # 確保時間為整數
  scatter_data['座標-X'] = scatter_data['座標-X'].astype(float)
  scatter_data['座標-Y'] = scatter_data['座標-Y'].astype(float)
  minx = scatter_data.loc[:, '座標-X'].min()
  maxx = scatter_data.loc[:, '座標-X'].max()
  miny = scatter_data.loc[:, '座標-Y'].min()
  maxy = scatter_data.loc[:, '座標-Y'].max()
  
  print(f"{minx} {maxx} {miny} {maxy}")
  # 繪製散點圖
  img= plt.imread('./zhongsan.jpg')
  
  fig, ax = plt.subplots()
  ax.imshow(img, extent=[minx-0.01, maxx+0.01, miny-0.01, maxy+0.01])
  scatter = plt.scatter(
      scatter_data['座標-X'], scatter_data['座標-Y'],
      c=scatter_data['發生時-Hours'], cmap='viridis', alpha=0.7, edgecolors='k'
  )
  plt.colorbar(scatter, label='發生時-Hours')  #點的顏色代表時間
  plt.title('肇事時間與地點分佈', fontsize=16)
  plt.xlabel('座標-X', fontsize=14)
  plt.ylabel('座標-Y', fontsize=14)
  plt.xlim(minx-0.01, maxx+0.01)
  plt.ylim(miny-0.01, maxy+0.01)
  #plt.xticks(np.arange(float(minx), float(maxx), 0.01).tolist())
  #plt.yticks(np.arange(float(miny), float(maxy), 0.01).tolist())
  plt.grid(alpha=0.3)
  plt.show()
  '''