"""
測定データをまとめる
"""

import datetime
import glob
import os
import sys
import subprocess
from tkinter import Tk
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy import optimize as opt
from sklearn import linear_model


INPUT_PATH = None
DEST_PATH = None


def browse() -> str:
    """
    フォルダの参照
    """
    global INPUT_PATH # pylint: disable=global-statement

    if INPUT_PATH is None:
        root = Tk()
        root.withdraw()
        current_directory = os.path.abspath(os.path.dirname(__file__))
        INPUT_PATH = askdirectory(initialdir=current_directory)
        if INPUT_PATH == "":
            print("キャンセルされました。")
            sys.exit(1)

    return INPUT_PATH


def get_curtime() -> str:
    """
    現在時刻を入手
    """
    times = datetime.datetime.now()
    time_ymd = times.strftime('%y%m%d-%H%M')

    return time_ymd


# pylint: disable=invalid-name
def write_linear_graph(lines, fname):
    """
    線形回帰から傾きと切片を求め、グラフを出力
    """
    # 線形モデル
    model = linear_model.LinearRegression()

    voltage_s = []
    current_s = []
    for _param in lines:
        voltage, current = _param.split(',')
        voltage_s.append(int(voltage))
        current_s.append(float(current))
    voltage_s = np.array(voltage_s).reshape(-1,1)
    current_s = np.array(current_s, dtype=np.float32).reshape(-1,1)
    model.fit(voltage_s, current_s)

    coef, intercept = model.coef_[0][0], model.intercept_[0]

    v_oc = round(intercept/coef, 3)
    i_sc = round(-1e+12*(intercept), 3)
    cond = round(1e+12*coef, 3)

    liner_image_path = os.path.join(
        DEST_PATH,
        os.path.splitext(fname)[0]+'_liner.png')
    plt.title(f"Slope:{cond} pS, Intercept:{i_sc} pA")
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.plot(voltage_s, current_s, marker='o', linestyle='None', color='black')
    plt.plot(voltage_s, model.predict(voltage_s), linestyle='solid', color='red')
    plt.savefig(liner_image_path)
    plt.close()

    return v_oc, i_sc, cond


def polar_formura(x, a, b, c):
    """
    偏向角特性
    卒論3章を参照
    """
    return a * np.sin(np.pi*(x/90. + b/180.)) + c


def write_polarization_graph(data):
    """
    偏向角-開放端電圧、偏向角-短絡電流のグラフを書き出す
    """

    # 分割
    angles, voltages, currents = data

    # カーブフィッティング
    # polar_formuraの変数(a, b, c)を求める
    sv_params = opt.curve_fit(polar_formura, angles, voltages)[0][0:3]
    sa_params = opt.curve_fit(polar_formura, angles, currents)[0][0:3]

    # 曲線を書く用のx軸データ
    polar_angles = np.arange(angles[0], angles[-1], step=5)

    # 曲線用のデータを作成
    pred_y = [[],[]]
    for angle in polar_angles:
        pred_y[0] += [polar_formura(angle, *sv_params)]
        pred_y[1] += [polar_formura(angle, *sa_params)]

    ##################################### グラフ書き出し #####################################
    fig, (graph_l, graph_r) = plt.subplots(ncols=2, figsize=(10,4))

    def write_data(y_data, txtpath):
        with open(txtpath, 'w') as f:
            f.writelines([
                f'{angles[i]}\t{y_data[i]}\n' for i in range(len(angles))])

    def plot_voltage():
        """
        偏向角-開放端電圧特性
        """
        graph_l.plot(angles, voltages, linestyle='None', marker='o', color='green')
        graph_l.plot(polar_angles, pred_y[0])
        graph_l.set_title('Voc-φ')
        graph_l.set_xlabel('Angle of light polarization φ (deg)')
        graph_l.set_ylabel('Open circuit voltage (V)')

        # ngraph用のデータを書き出す
        write_data(voltages, os.path.join(DEST_PATH, 'ngraph_Voc.txt'))

    def plot_current():
        """
        偏向角-短絡電流特性
        """
        graph_r.plot(angles, currents, linestyle='None', marker='o', color='blue')
        graph_r.plot(polar_angles, pred_y[1])
        graph_r.set_title('Isc-φ')
        graph_r.set_xlabel('Angle of light polarization φ (deg)')
        graph_r.set_ylabel('Short circuit current (pA)')

        write_data(currents, os.path.join(DEST_PATH, 'ngraph_Isc.txt'))

    plot_voltage()
    plot_current()
    fig.savefig(os.path.join(DEST_PATH, 'polarization_Voc_Isc.png'))
    plt.close()
    #########################################################################################


def write_to_excel(data):
    """
    エクセルデータを出力(使用module:openpyxl)
    """

    # pylint: disable=abstract-class-instantiated
    # pylint: disable=unsubscriptable-object
    output_file = os.path.join(DEST_PATH, 'result.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        columns = [
            'Polarizing plate angle (deg)',
            'Polarization angle (deg)',
            'Voltage (V)',
            'Current (pA)',
            'Conductance (pS)'
            ]
        df_data = pd.DataFrame(data, columns=columns)
        df_data.to_excel(writer, sheet_name='result', engine='openpyxl')

        # エクセルのコラム幅を変える
        worksheet = writer.book['result']
        for i, cell in enumerate(['B', 'C', 'D', 'E', 'F']):
            worksheet.column_dimensions[cell].width = len(columns[i])


def main():
    """
    データを作成(フォーマットは林本さんの形式)
    """
    global DEST_PATH # pylint: disable=global-statement

    DEST_PATH = os.path.join(browse(), get_curtime()+'_result')
    print(browse())
    os.makedirs(DEST_PATH, exist_ok=True)
    print("プログレスバーが表示されないときは何かのキーを押してください。")

    paths = [p for p in glob.glob(browse()+'/**', recursive=True)
        if os.path.splitext(p)[1].lower()=='.dat']

    error_log = []
    nums = {}
    for _path in paths:
        try:
            num = float(
                os.path.splitext(os.path.split(_path)[1])[0])
        except ValueError:
            error_log.append(f"Invalid file name : {os.path.abspath(_path)}\n")
            continue
        nums[_path] = num

    if len(nums) == 0:
        error_log.append(".DAT file could not be loaded.")
    if len(error_log) > 0:
        output_file = os.path.join(DEST_PATH, 'Error.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(error_log)
            if len(nums) == 0:
                sys.exit(1)

    nums = sorted(nums.items(), key=lambda x: x[1])

    data = []
    polar_angle = 0.
    polarizations = [[],[],[]]
    for _nums in tqdm.tqdm(nums):
        with open(_nums[0], 'r') as f:
            v_oc, i_sc, cond = write_linear_graph(
                f.readlines()[13:],
                os.path.split(_nums[0])[-1])
        data += [[_nums[1], polar_angle, v_oc, i_sc, cond]]
        polarizations[0] += [_nums[1]]
        polarizations[1] += [v_oc]
        polarizations[2] += [i_sc]
        polar_angle += 11.25

    write_polarization_graph(polarizations)
    write_to_excel(data)


if __name__=='__main__':
    main()

    print("Done!")
    OPEN_PATH = DEST_PATH.replace('/', '\\')
    _ = subprocess.run(f"explorer {OPEN_PATH}", shell=True)
