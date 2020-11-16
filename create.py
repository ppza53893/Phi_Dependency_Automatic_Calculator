"""
測定データをまとめる
"""

import datetime
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from sklearn import linear_model


def get_curtime() -> str:
  times = datetime.datetime.now()
  time_ymd = times.strftime('%y%m%d-%H%M')

  return time_ymd


def output_linear_params(lines, output_path, fname, has_plt=False):
  """
  最小二乗法を計算する
  """
  model = linear_model.LinearRegression()

  voltage_s = []
  ampere_s = []
  for _param in lines:
    voltage, ampere = _param.split(',')
    voltage_s.append(int(voltage))
    ampere_s.append(float(ampere))
  voltage_s = np.array(voltage_s).reshape(-1,1)
  ampere_s = np.array(ampere_s, dtype=np.float32).reshape(-1,1)
  model.fit(voltage_s, ampere_s)

  coef, intercept = model.coef_[0][0], model.intercept_[0]

  if has_plt:
    liner_image_path = os.path.join(
      output_path,
      os.path.splitext(fname)[0]+'_liner.png')
    plt.title(
      f"Slope:{round(1e+12*coef, 3)} pS,\
        Intercept:{round(-1e+12*(intercept), 3)} pA")
    plt.xlabel('Voltage [V]')
    plt.ylabel('Ampere [A]')
    plt.plot(voltage_s, ampere_s, marker='o', color='black')
    plt.plot(voltage_s, model.predict(voltage_s), linestyle='solid', color='red')
    plt.savefig(liner_image_path)
    plt.close()

  return coef, intercept


def create(data_path='target'):
  """
  データを作成(フォーマットは林本さんの形式)
  """
  # define
  error_log = []
  output_f = get_curtime()+'_result'
  os.makedirs(output_f, exist_ok=True)
  output_name = os.path.join(output_f, 'result.xlsx')
  error_file_name = os.path.join(output_f, 'Error.txt')

  paths = [p for p in glob.glob(data_path+'/**', recursive=True)
    if os.path.splitext(p)[1].lower()=='.dat']

  has_error = False
  nums = {}
  for _path in paths:
    try:
      fname = os.path.split(_path)[1]
      num = float(os.path.splitext(fname)[0])
    except ValueError:
      error_log.append(f"Invalid file name : {os.path.abspath(_path)}\n")
      continue
    nums[_path] = num
  if len(nums) == 0:
    error_log.append(".dat file could not be loaded.")
    has_error = True
  if len(error_log) > 0:
    with open(error_file_name, 'w', encoding='utf-8') as f:
      f.writelines(error_log)
    if has_error:
      sys.exit(1)
  
  nums = sorted(nums.items(), key=lambda x: x[1])

  data = []
  angle_zero_base = 0.
  zerobase_value = round(180/(len(nums)-1), 1)
  plot_image = len(sys.argv)>1
  for _nums in tqdm.tqdm(nums):
    with open(_nums[0], 'r') as f:
      lines = f.readlines()[13:]
    coef, intercept = output_linear_params(
      lines,
      output_f,
      os.path.split(_nums[0])[-1],
      plot_image)
    data += [[
      _nums[1],
      angle_zero_base,
      round(intercept/coef, 3),
      round(-1e+12*(intercept), 3),
      round(1e+12*coef, 3)
      ]]
    angle_zero_base += zerobase_value

  columns = [
    'angle',
    'angle(zero_base)',
    'voltage [V]',
    'ampere [pS]',
    'conductance [pS]']
  df_data = pd.DataFrame(data, columns = columns)

  # pylint: disable=abstract-class-instantiated
  # pylint: disable=unsubscriptable-object
  with pd.ExcelWriter(output_name) as writer:
    df_data.to_excel(writer, sheet_name='result', engine='openpyxl')
    worksheet = writer.book['result']
    worksheet.column_dimensions['C'].width = 18
    worksheet.column_dimensions['D'].width = 15
    worksheet.column_dimensions['E'].width = 18
    worksheet.column_dimensions['F'].width = 18


if __name__=='__main__':
  create()
