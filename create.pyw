"""
測定データをまとめる
"""
# pylint: disable=W1510
import datetime
import glob
import msvcrt
import os
import shutil
import sys
import subprocess
_ = subprocess.run("cls", shell=True)
_ = subprocess.run("title Polarization Automatic Tool", shell=True)
if not __debug__:
    print("デバッグモード ON")
CMD_TERMINAL_LENGTH = shutil.get_terminal_size().columns
print('#'*CMD_TERMINAL_LENGTH)
print("Polarization Automatic Tool ver 1.52 (Update : 2020/12/05)")
print(">>> 使用方法は「How to use.txt」を見てください。")
print(">>> 初回起動は少しロードが遅くなります。")
print(">>> ブラウズ画面やプログレスバーが表示されないときは何かのキーを押してください。")
print('#'*CMD_TERMINAL_LENGTH)
sys.stdout.write("\n")

# pylint: disable=wrong-import-position
from tkinter import Tk
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy import optimize as opt
from sklearn import linear_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.ngraph import NgraphWriter
# pylint: enable=wrong-import-position


DIR_SELECTED = False
INPUT_PATH = None
DEST_PATH = None


##################################### 変更場所 #####################################
DAT_READ_START = 13 # データ(.DAT)の読み込み開始位置
ANGLE_ADD = 22.5 # 回転角のステップを変える場合はここを変える
###################################################################################


# pylint: disable=invalid-name
def browse() -> str:
    """
    フォルダの参照
    """
    global INPUT_PATH # pylint: disable=global-statement
    global DIR_SELECTED # pylint: disable=global-statement

    if not DIR_SELECTED:
        abs_path = os.path.abspath(os.path.dirname(__file__))
        find_path = os.path.join(abs_path, '.record')
        if os.path.exists(find_path):
            with open(find_path, 'r') as f:
                ask_dir = f.readline().rstrip()
                if not os.path.exists(ask_dir):
                    ask_dir = abs_path
        else:
            ask_dir = abs_path
        root = Tk()
        root.withdraw()
        INPUT_PATH = askdirectory(initialdir=ask_dir)
        if INPUT_PATH == "":
            print(">>> キャンセルされました。")
            sys.exit(0)
        DIR_SELECTED = True

    return INPUT_PATH


def get_curtime(for_folder=True) -> str:
    """
    現在時刻を入手
    """
    times = datetime.datetime.now()
    time_ymd = times.strftime('%y%m%d-%H%M') \
        if for_folder else times.strftime('%Y/%m/%d %H:%M')

    return time_ymd


def write_pathlog():
    """
    読み込んだフォルダパスを書き込み(次回以降の参照先にするため)
    """
    record_file = '.record'
    file_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), record_file)
    with open(file_path, 'w') as f:
        f.write(INPUT_PATH + '\n')
        f.write('Last used:'+ get_curtime(False))


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
    理論式(deg)
    林本さん卒論3章を参照
    """
    ##################################### 変更場所 #####################################
    # 理論式を変える場合はここを変える
    return a * np.sin(np.pi*(x/90. + b/180.)) + c
    ###################################################################################


def calc_y_scale(y, yp):
    """
    y軸のスケール幅を設定する
    参考 : https://imagingsolution.net/program/autocalcgraphstep/
    """
    _y = max(
            abs(max(y)),
            abs(min(y)),
            abs(max(yp)),
            abs(min(yp)))
    ymax, npow = f"{_y:e}".split("e")
    ymax = float(ymax)
    npow = abs(int(npow))

    # 軸幅の設定
    if ymax < 1.5:
        step = 0.5*10**npow
    elif ymax < 3.5:
        step = 1.0*10**npow
    elif ymax <= 5.0:
        step = 2.0*10**npow
    else:
        step = 5.0*10**npow

    def rangeset(value):
        n = abs(value)
        ret = step
        i = 2
        while n > ret:
            ret = step*i
            i += 1
        return ret if value >=0 else -ret

    # 上下の高さを統一する
    y_min = rangeset(min(min(y), min(yp)))
    y_max = rangeset(max(max(yp), max(yp)))
    y_min = max(abs(y_max), abs(y_min)) * abs(y_min) / y_min
    y_max = max(abs(y_max), abs(y_min)) * abs(y_max) / y_max
    if not __debug__:
        print("(min, max, step) = ", y_min, y_max, step)
    return y_min, y_max, step


def write_ngp_data(angles, y_data, y_pred, s_params, txtpath, mode):
    """
    Ngraphデータを生成
    """
    with open(txtpath, 'w') as f:
        f.writelines([
            f'{angles[i]}\t{y_data[i]}\n' for i in range(len(angles))])

    y_scales = calc_y_scale(y_data, y_pred)
    directory = os.path.join('.', os.path.split(txtpath)[1]).replace("\\", '/')

    ngp = NgraphWriter(mode)
    ngp.write([
        [0,'axis name:fX1', 'min', angles[0]],
        [0,'axis name:fX1', 'max', angles[-1]],
        [0,'axis name:fX1', 'inc', ANGLE_ADD*2],
        [1,'axis name:fY1', 'min', y_scales[0]],
        [1,'axis name:fY1', 'max', y_scales[1]],
        [1,'axis name:fY1', 'inc', y_scales[2]],
        [5, 'file', 'file', f"'{directory}'"],
        ##################################### 変更場所 #####################################
        # 理論式を変える場合はここも変える
        # 書き方はngpファイル参照(fit::equation)
        [6,'fit','equation',
            f"'{s_params[0]}*"\
                f"SIN(PI*(X/90+{s_params[1]}/180))+{s_params[2]}'"],
        ###################################################################################
        [7, 'file', 'file', f"'{directory}'"],
        ])
    ngp.out(txtpath.replace('.txt', '.ngp'))


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

    def plot_voltage():
        """
        偏向角-開放端電圧特性
        """
        graph_l.plot(angles, voltages, linestyle='None', marker='o', color='green')
        graph_l.plot(polar_angles, pred_y[0],
            label=f'Voc={sv_params[0]:.1f}'\
                f'sin(2φ{sv_params[1]*3.1415926535/180:+.1f}){sv_params[2]:+.1f}')
        if not __debug__:
            print('voc:',sv_params)
        ##################################### 変更場所 #####################################
        # 特性グラフの軸の名前
        graph_l.set_title('Voc-φ')
        graph_l.set_xlabel('Angle of light polarization φ (deg)')
        ###################################################################################
        graph_l.set_ylabel('Open circuit voltage (V)')
        graph_l.set_xlim([angles[0], angles[-1]])
        graph_l.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)

        # ngraph用のデータを書き出す
        write_ngp_data(
            angles,
            voltages,
            pred_y[0],
            sv_params,
            os.path.join(DEST_PATH, 'Voc_phi.txt'),
            'v')

    def plot_current():
        """
        偏向角-短絡電流特性
        """
        graph_r.plot(angles, currents, linestyle='None', marker='o', color='blue')
        graph_r.plot(polar_angles, pred_y[1],
            label=f'Isc={sa_params[0]:.1f}'\
                f'sin(2φ{sa_params[1]*3.1415926535/180:+.1f}){sa_params[2]:+.1f}')
        if not __debug__:
            print('isc:',sa_params)
        ##################################### 変更場所 #####################################
        # 特性グラフの軸の名前
        graph_r.set_title('Isc-φ')
        graph_r.set_xlabel('Angle of light polarization φ (deg)')
        ###################################################################################
        graph_r.set_ylabel('Short circuit current (pA)')
        graph_r.set_xlim([angles[0], angles[-1]])
        graph_r.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)

        write_ngp_data(
            angles,
            currents,
            pred_y[1],
            sa_params,
            os.path.join(DEST_PATH, 'Isc_phi.txt'),
            'i')

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
    print('>>> パス : '+browse())

    error_log = []
    nums = {}
    for _path in glob.glob(browse()+'/*'):
        if not os.path.isfile(_path):
            continue
        try:
            num = float(os.path.split(_path)[1])
        except ValueError:
            if os.path.splitext(os.path.split(_path)[1])[1].lower() != '.dat':
                continue
            try:
                num = float(
                    os.path.splitext(os.path.split(_path)[1])[0])
            except ValueError as e:
                error_log.append(str(e) + '\n')
                error_log.append(f"Invalid file name : {os.path.abspath(_path)}\n")
                continue
        nums[_path] = num
    if len(nums) == 0:
        error_log.append("file could not be loaded.")
        print(">>> エラー : 有効なファイルがありませんでした。\n>>> ほかのデータを参照しますか? (y/n):", end='')
        return

    nums = sorted(nums.items(), key=lambda x: x[1])
    os.makedirs(DEST_PATH, exist_ok=True)
    if len(error_log) > 0:
        output_file = os.path.join(DEST_PATH, 'Error.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(error_log)

    data = []
    polar_angle = 0.
    polarizations = [[],[],[]]
    for _nums in tqdm.tqdm(nums):
        with open(_nums[0], 'r') as f:
            v_oc, i_sc, cond = write_linear_graph(
                dat_separate(f.readlines()[DAT_READ_START:]),
                os.path.split(_nums[0])[-1])

        data += [[_nums[1], polar_angle, v_oc, -i_sc, cond]]
        polarizations[0] += [polar_angle]
        polarizations[1] += [v_oc]
        polarizations[2] += [-i_sc]
        polar_angle += ANGLE_ADD
    write_polarization_graph(polarizations)
    write_to_excel(data)
    write_pathlog()
    if not __debug__:
        sys.exit(0)
    print(">>> 完了!\n>>> 続行しますか? (y/n):", end='')

    # ファイルを開く
    if os.name == 'nt':
        DEST_PATH = DEST_PATH.replace('/', '\\')
        if __debug__:
            _ = subprocess.run(f"explorer {DEST_PATH}", shell=True)


def reset():
    """
    グローバル変数のリセット
    """
    # pylint: disable=global-statement
    global DIR_SELECTED
    global DEST_PATH
    # pylint: enable=global-statement

    DIR_SELECTED = False
    DEST_PATH = None


if __name__=='__main__':
    main()

    while True:
        key = ord(msvcrt.getch())
        sys.stdout.write(f"{chr(key)}")
        if key==3 or chr(key).lower()=='n':
            print()
            sys.exit(0)
        elif chr(key).lower()=='y':
            sys.stdout.write("\n")
            reset()
            main()
        else:
            print_write = "\n>>> 無効なキーです。「y」か「n」を押してください:"
            if key == 0x1B:
                sys.stdout.write('0')
            sys.stdout.write(print_write)
