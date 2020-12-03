"""
測定データをまとめる
"""

import datetime
import glob
import msvcrt
import os
import sys
import subprocess
_ = subprocess.run("title Polarization Automatic Tool", shell=True)
print("Polarization Automatic Tool ver 1.3 (2020/12/03)")
print("・使用方法は「How to use.txt」を見てください。")
print("・初回起動は少しロードが遅くなります。")
print("・ブラウズ画面やプログレスバーが表示されないときは何かのキーを押してください。\n")

# pylint: disable=wrong-import-position
from tkinter import Tk
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy import optimize as opt
from sklearn import linear_model
# pylint: enable=wrong-import-position


INPUT_PATH = None
DEST_PATH = None
DAT_READ_START = 13


##################################### 変更場所 #####################################
ANGLE_ADD = 22.5 # 回転角のステップを変える場合はここを変える
###################################################################################


# pylint: disable=invalid-name
def browse() -> str:
    """
    フォルダの参照
    """
    global INPUT_PATH # pylint: disable=global-statement

    if INPUT_PATH is None:
        root = Tk()
        root.withdraw()
        #INPUT_PATH = askdirectory(
        #    initialdir="C:\\Users\\Owner\\Documents\\DATA")
        INPUT_PATH = askdirectory(
            initialdir=os.path.abspath(os.path.dirname(__file__)))
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


def calc_linears(x, y) -> list:
    """
    線形回帰
    """

    model = linear_model.LinearRegression()
    x = np.array(x, dtype=np.float32).reshape(-1,1)
    y = np.array(y, dtype=np.float32).reshape(-1,1)
    model.fit(x, y)

    return model, model.score(x, y)


def dat_separate(lines):
    """
    DATファイルから電圧と電流のデータを取得
    """
    sep_lines = [x.split(',') for x in lines]
    v_s = [float(x[0]) for x in sep_lines]
    i_s = [float(x[1]) for x in sep_lines]
    return [v_s, i_s]


def get_best_model(data):
    """
    決定係数がもっともよいモデルを取得し、予測値を取得
    """

    count = len(data[0])>>1
    models = []
    scores = np.zeros((count, ), dtype=np.float32)
    for i in range(count):
        model = linear_model.LinearRegression()
        x = np.array(data[0][i:], dtype=np.float32).reshape(-1,1)
        y = np.array(data[1][i:], dtype=np.float32).reshape(-1,1)
        model.fit(x, y)

        models.append(model)
        scores[i] = model.score(x,y)

    model = models[np.argmax(scores)]
    x = np.array(data[0], dtype=np.float32).reshape(-1,1)
    y = model.predict(x)

    return model, x, y


def write_linear_graph(data, fname):
    """
    線形回帰から傾きと切片を求め、グラフを出力
    """
    # 線形モデル

    model, pred_x, pred_y = get_best_model(data)

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
    plt.plot(data[0], data[1], marker='o', linestyle='None', color='black')
    plt.plot(pred_x, pred_y, linestyle='solid', color='red')
    plt.savefig(liner_image_path)
    plt.close()

    return v_oc, i_sc, cond


#___################################# 変更場所 #####################################
def formula(x, a, b, c):
    # ↑ 理論式の変数の個数に合わせて変数を追加する
    ###############################################################################
    """
    理論式
    林本さん卒論3章を参照
    """
    ##################################### 変更場所 #####################################
    # 理論式を変える場合はここを変える
    return a * np.sin(np.pi*(x/90. + b/180.)) + c
    ###################################################################################


def write_polarization_graph(data):
    """
    偏向角-開放端電圧、偏向角-短絡電流のグラフを書き出す
    """

    # 分割
    angles, voltages, currents = data

    # カーブフィッティング
    # formulaの変数を求める
    sv_params = opt.curve_fit(formula, angles, voltages)[0]
    sa_params = opt.curve_fit(formula, angles, currents)[0]

    # 曲線を書く用のx軸データ
    polar_angles = np.arange(angles[0], angles[-1], step=2)

    # 曲線用のデータを作成
    pred_y = [[],[]]
    for angle in polar_angles:
        pred_y[0] += [formula(angle, *sv_params)]
        pred_y[1] += [formula(angle, *sa_params)]

    # グラフ書き出し
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
        graph_l.plot(polar_angles, pred_y[0],
            label=f'Voc={round(sv_params[0], 1)}sin(2φ{round(sv_params[1]*3.1415926535/180,1):+}){round(sv_params[2],1):+}')
        ##################################### 変更場所 #####################################
        # 特性グラフの軸の名前
        graph_l.set_title('Voc-φ')
        graph_l.set_xlabel('Angle of light polarization φ (deg)')
        ###################################################################################
        graph_l.set_ylabel('Open circuit voltage (V)')
        graph_l.set_xlim([angles[0], angles[-1]])
        graph_l.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)

        # ngraph用のデータを書き出す
        write_data(voltages, os.path.join(DEST_PATH, 'ngraph_Voc.txt'))

    def plot_current():
        """
        偏向角-短絡電流特性
        """
        graph_r.plot(angles, currents, linestyle='None', marker='o', color='blue')
        graph_r.plot(polar_angles, pred_y[1],
            label=f"Isc={round(sa_params[0], 1)}sin(2φ{round(sa_params[1]*3.1415926535/180,1):+}){round(sa_params[2],1):+}")
        ##################################### 変更場所 #####################################
        # 特性グラフの軸の名前
        graph_r.set_title('Isc-φ')
        graph_r.set_xlabel('Angle of light polarization φ (deg)')
        ###################################################################################
        graph_r.set_ylabel('Short circuit current (pA)')
        graph_r.set_xlim([angles[0], angles[-1]])
        graph_r.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)

        write_data(currents, os.path.join(DEST_PATH, 'ngraph_Isc.txt'))

    plot_voltage()
    plot_current()
    fig.savefig(os.path.join(DEST_PATH, 'polarization_Voc_Isc.png'))
    plt.close()


def write_to_excel(data):
    """
    エクセルデータを出力(使用module:openpyxl)
    """

    # pylint: disable=abstract-class-instantiated
    # pylint: disable=unsubscriptable-object
    output_file = os.path.join(DEST_PATH, 'result.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        ##################################### 変更場所 #####################################
        # excelの内容
        columns = [
            'Polarizing plate angle (deg)',
            'Polarization angle (deg)',
            'Voltage (V)',
            'Current (pA)',
            'Conductance (pS)'
            ]
        ###################################################################################
        df_data = pd.DataFrame(data, columns=columns)
        df_data.to_excel(writer, sheet_name='result', engine='openpyxl')

        # エクセルのコラム幅を変える
        worksheet = writer.book['result']
        for i, cell in enumerate(['B', 'C', 'D', 'E', 'F']):
            worksheet.column_dimensions[cell].width = len(columns[i])+1


def main():
    """
    データを作成(フォーマットは林本さんの形式)
    """
    global DEST_PATH # pylint: disable=global-statement

    DEST_PATH = os.path.join(browse(), get_curtime()+'_result')
    
    print(browse())
    os.makedirs(DEST_PATH, exist_ok=True)

    paths = []
    for _path in glob.glob(browse()+'/*'):
        try:
            num = float(os.path.split(_path)[1])
        except ValueError:
            if os.path.splitext(_path)[1].lower() !='.dat':
                continue
        paths.append(_path)

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
                print("エラー:有効なファイルがありませんでした。\nほかのデータを参照しますか? (y/n):", end='')
                return

    nums = sorted(nums.items(), key=lambda x: x[1])

    data = []
    polar_angle = 0.
    polarizations = [[],[],[]]
    for _nums in tqdm.tqdm(nums):
        with open(_nums[0], 'r') as f:
            v_oc, i_sc, cond = write_linear_graph(
                dat_separate(f.readlines()[DAT_READ_START:]),
                os.path.split(_nums[0])[-1])

        data += [[_nums[1], polar_angle, v_oc, i_sc, cond]]
        polarizations[0] += [polar_angle]
        polarizations[1] += [v_oc]
        polarizations[2] += [i_sc]

        polar_angle += ANGLE_ADD

    write_polarization_graph(polarizations)
    write_to_excel(data)
    print("完了!\n続行しますか? (y/n):", end='')

    # ファイルを開く
    if os.name == 'nt':
        DEST_PATH = DEST_PATH.replace('/', '\\')
        _ = subprocess.run(f"explorer {DEST_PATH}", shell=True)


def reset():
    """
    グローバル変数のリセット
    """
    # pylint: disable=global-statement
    global INPUT_PATH
    global DEST_PATH
    # pylint: enable=global-statement

    INPUT_PATH = None
    DEST_PATH = None

    print()


if __name__=='__main__':
    main()

    while True:
        key = ord(msvcrt.getch())
        sys.stdout.write(f"{chr(key)}")
        if key==3 or chr(key).lower()=='n':
            break
        elif chr(key).lower()=='y':
            reset()
            main()
        else:
            sys.stdout.write("\n無効なキーです。「y」か「n」を押してください:")
            continue
    print()

            
        
