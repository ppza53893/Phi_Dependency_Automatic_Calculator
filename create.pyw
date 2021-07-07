"""
測定データをまとめる
"""
# pylint: disable=W1510
import datetime
import math
import msvcrt
import os
import shutil
import sys
import subprocess


def call_cmd(command:str, **kwargs):
    """
    cmdコマンドを実行
    """
    if isinstance(command, (str, list, tuple)):
        if 'shell' not in kwargs:
            kwargs.update({'shell':True})
        _ = subprocess.run(command, **kwargs)
    else:
        print(">>> [ERROR] 有効なコマンドではありません : " + str(command))
call_cmd("cls")
#call_cmd("color 0B")
call_cmd("title Phi Dependency Automatic Calculator")

# コマンドライン幅
CMD_TERMINAL_LENGTH = shutil.get_terminal_size().columns
# カレントディレクトリ
CWD = os.path.dirname(os.path.abspath(__file__))
# ngraph.pyの参照用
sys.path.append(CWD)


# デバッグモード(内部テスト用)
# 変更禁止
IS_DEBUG = not __debug__
if IS_DEBUG:
    print("デバッグモード ON")

# はじめの画面
print('\033[38;5;010m#'*CMD_TERMINAL_LENGTH)
print("Phi Dependency Automatic Calculator v2.00 (Update : 2021/01/24, Created by OHASHI)")
print(">>> \033[38;5;011m使用方法は「How to use.txt」を見てください。\033[38;5;010m")
print(">>> \033[38;5;011mフォルダ参照を使用する場合、初回は少しロードが遅くなります。\033[38;5;010m")
print(">>> \033[38;5;011mブラウズ画面やプログレスバーが表示されないときは何かのキーを押してください。\033[38;5;010m")
print('#'*CMD_TERMINAL_LENGTH)
sys.stdout.write("\n")

FOR_VEE = len(sys.argv) > 1
if FOR_VEE:
    print(">>>\033[38;5;011m [LOG] CMD引数を確認しました。\033[38;5;010m")


# pylint: disable=wrong-import-position
from glob import iglob
from glob import escape as escape_chr
from statistics import mode
from tkinter import Tk
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy import optimize as opt
from sklearn import linear_model
from time import sleep

from lib.ngraph import NgraphWriter
from lib.si import get_prefix, get_best_prefix
# pylint: enable=wrong-import-position


# 下のは内部で変わるので触らないこと。
#_____________ここから_____________
# ブラウズ用 すでにフォルダは選択したか?
DIR_SELECTED = False
# ブラウズ後のディレクトリパス
INPUT_PATH = None
# データ出力パス
DEST_PATH = None
# Ngrapghモード
NGP_MODE = None
# prefix
PREFIX_V = []
PREFIX_I = []
PREFIX_G = []
# excel(I-V) plot
IV_DATA = {}
#______________ここまで_____________

# plt initialize
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

##################################### 変更場所 #####################################
ANGLE_ADD = 22.5 # 回転角のステップを変える場合はここを変える
###################################################################################


# pylint: disable=invalid-name
def browse():
    """
    フォルダの参照
    """
    global INPUT_PATH # pylint: disable=global-statement
    global DIR_SELECTED # pylint: disable=global-statement

    if FOR_VEE:
        INPUT_PATH = sys.argv[1]
        if len(sys.argv) > 2:
            for i in range(2, len(sys.argv)):
               INPUT_PATH += ' ' + sys.argv[i]
        DIR_SELECTED = True
    elif not DIR_SELECTED:
        # 前回実行したログはあるか?
        find_path = os.path.join(CWD, '.record')
        if os.path.exists(find_path):
            with open(find_path, 'r') as f:
                ask_dir = f.readline().rstrip()
                if not os.path.exists(ask_dir):
                    ask_dir = CWD
        else:
            ask_dir = CWD

        # ブラウジング
        root = Tk()
        root.withdraw()
        INPUT_PATH = askdirectory(initialdir=ask_dir)
        if INPUT_PATH == "":
            print(">>> キャンセルされました。\033[0m")
            sys.exit(0)
        DIR_SELECTED = True


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
    file_path = os.path.join(CWD, record_file)
    with open(file_path, 'w') as f:
        f.write(INPUT_PATH + '\n')
        f.write('Last used:'+ get_curtime(False))


def convert_to_prefix_values(data, x_exp, y_exp):
    """
    データを接頭辞に合わせる
    """

    if isinstance(data, list):
        _data = np.array(data)
    else:
        _data = data.copy()
    
    _data[0] = _data[0]/10**x_exp
    _data[1] = _data[1]/10**y_exp

    return _data


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
    y = model.predict(x).reshape(1,-1)[0]

    returns = np.vstack([x.reshape(1,-1)[0], y])

    return model, returns


def write_linear_graph(data, fname):
    """
    線形回帰から傾きと切片を求め、グラフを出力
    """
    # 線形モデル
    global PREFIX_I, PREFIX_V, PREFIX_G, IV_DATA

    model, pred = get_best_model(data)
    v_pref = get_prefix(pred[0])
    i_pref = get_prefix(pred[1])

    coef, intercept = model.coef_[0][0], model.intercept_[0]

    cf_prefixes = get_prefix(coef)
    ic_prefixes = get_prefix(intercept)

    i_sc = intercept/10**ic_prefixes[1]
    cond = coef/10**cf_prefixes[1]


    liner_image_path = os.path.join(
        DEST_PATH, 'I-V characteristics',
        os.path.splitext(fname)[0]+'_plot.png')

    v_prefixes = get_prefix(intercept/coef)
    fix_data = convert_to_prefix_values(data, v_pref[1], i_pref[1])
    fix_pred = convert_to_prefix_values(pred, v_pref[1], i_pref[1])
    plt.title(f"Slope:{cond:.3f} {cf_prefixes[0]}S, Intercept:{i_sc:.3f} {ic_prefixes[0]}A")
    plt.xlabel(f'Voltage ({v_pref[0]}V)')
    plt.ylabel(f'Current ({i_pref[0]}A)')
    plt.plot(fix_data[0], fix_data[1], marker='o', linestyle='None', color='black')
    plt.plot(fix_pred[0], fix_pred[1], linestyle='solid', color='red')
    plt.savefig(liner_image_path)
    plt.close()

    PREFIX_I.append(ic_prefixes[1])
    PREFIX_V.append(v_prefixes[1])
    PREFIX_G.append(cf_prefixes[1])
    IV_DATA[os.path.splitext(fname)[0]] = {
    'prefix' : [v_pref, i_pref],
    'data' : np.vstack([fix_data[0], fix_data[1], fix_pred[1]]).T
    }
    return intercept/coef, -intercept, coef # 短絡電流はマイナスにしておく


#___################################# 変更場所 #####################################
def formula(x, a, b, c):
    # ↑ 理論式の変数の個数に合わせて変数を追加する
    ###############################################################################
    """
    理論式(deg)
    林本さん卒論3章 or 卒研フォルダを参照
    """
    ##################################### 変更場所 #####################################
    # 理論式を変える場合はここを変える
    # 現在は(001)配向のBFOの偏光角依存性
    return a * np.sin(np.pi*(x/90. + b/180.)) + c
    ###################################################################################


def calc_y_bars(plt_ax):
    """
    y軸のスケール幅を設定する
    """

    act_bool = plt_ax[0]*plt_ax[1] > 0 # 全て負 or 正
    yma = max(abs(plt_ax[0]), abs(plt_ax[1]))
    ymm = min(abs(plt_ax[0]), abs(plt_ax[1]))
    if IS_DEBUG:
        print(plt_ax, yma, ymm)
    if act_bool:
        # 基点を調整する
        ref_gap = (yma+ymm)/2
        yma -= ref_gap
        ymm -= ref_gap
    else:
        ref_gap = 0.
        ymm = -ymm
    npow = round(math.log10(yma))
    # 軸幅の設定
    if yma/10**npow < 1.5:
        step = 0.5*10**npow
    elif yma/10**npow < 3.5:
        step = 1.0*10**npow
    elif yma/10**npow <= 5.0:
        step = 2.0*10**npow
    else:
        step = 5.0*10**npow

    if step < 0.1:
        while step < 0.1:
            # 別に問題ではないが、ステップ値が小さすぎるとngraphの軸目盛が表示されない
            step *= 2

    def rangeset(value):
        if value == 0:
            return step
        n = abs(value)
        ret = step
        i = 2
        while n > ret:
            ret = step*i
            i += 1
        return ret if value >=0 else -ret

    if act_bool:
        # 基点をもとの場所に戻す
        fix_gap = rangeset(ref_gap)
        sgn = -1 if max(plt_ax) < 0 else 1
        y_min = rangeset(ymm) + fix_gap * sgn
        y_max = rangeset(yma) + fix_gap * sgn
        # さらに補正
        while (
            y_min > min(plt_ax)
            or y_max < max(plt_ax)
        ):
            y_min -= step
            y_max += step
    else:
        y_min = rangeset(ymm)
        y_max = rangeset(yma)
        # 上下の高さを統一する
        y_min = max(abs(y_max), abs(y_min)) * abs(y_min) / y_min
        y_max = max(abs(y_max), abs(y_min)) * abs(y_max) / y_max

    check = int(abs((y_max-y_min)/step))
    if check == 2:
        # 軸数が2つの場合は4つにする
        step /= 2
    elif check > 6:
        # 軸数が6より大きい場合は半分にする
        step *= 2
    return [y_min, y_max, step]


def correct_x(x):
    """
    x軸の最大値を調節
    """
    if abs(x) % 45 != 0:
        ret = ANGLE_ADD*2
        i = 2
        while abs(x) > ret:
            ret = ANGLE_ADD*2*i
            i += 1
        x_max = ret if x >=0 else -ret
    else:
        x_max = x

    return x_max


def update_ngp_mode(mode):
    """
    NgraphWriterのモードを更新
    """
    global NGP_MODE # pylint: disable=global-statement
    NGP_MODE = mode


def write_ngp_data(angles, y_data, y_scales, s_params, txtpath, prefix):
    """
    Ngraphデータを生成
    """
    with open(txtpath, 'w') as f:
        f.writelines([
            f'{angles[i]}\t{y_data[i]}\n' for i in range(len(angles))])

    directory = os.path.join('.', os.path.split(txtpath)[1]).replace("\\", '/')

    def encode_escape(string):
        if string == '\xb5':
            string =  str(string.encode('unicode-escape')).split("'")[1][1:]
        return string

    ngp = NgraphWriter(NGP_MODE)
    _txt = r"'Short circuit current %f{HelvI}I%f{Helv}_sc@ "+f"({encode_escape(prefix)}A)'"\
        if NGP_MODE == 'i' else\
            r"'Open circuit voltage %f{HelvI}V%f{Helv}_oc@ "+f"({encode_escape(prefix)}V)'"
    ##################################### 変更場所 #####################################
    # ([])内の要素を追加できます。
    # 各リストの1番目はクラス番号(何番目のクラスか)、2番目はクラス名(ngpファイルの
    # 「new ...」)、 3番目はクラス内の変数名、4番目は代入する変数名です。
    # 文字を代入する場合は"''"のようにアポストロフィを入れないといけない場合と、
    # 入れなくても""か''でいいようなケースがあるので注意してください。
    # 間違っていると生成されたngpファイルを開いたらエラーが出ます。
    ngp.write([
        [0,'axis name:fX1', 'min', angles[0]],
        [0,'axis name:fX1', 'max', correct_x(angles[-1])],
        [0,'axis name:fX1', 'inc', ANGLE_ADD*2],
        [1,'axis name:fY1', 'min', y_scales[0]],
        [1,'axis name:fY1', 'max', y_scales[1]],
        [1,'axis name:fY1', 'inc', y_scales[2]],
        [5, 'file', 'file', f"'{directory}'"],
        # 理論式を変える場合はここも変える
        # 書き方はngpファイル参照(fit::equation)
        [6,'fit','equation',
            f"'{s_params[0]}*"\
                f"SIN(PI*(X/90+{s_params[1]}/180))+{s_params[2]}'"],
        [7, 'file', 'file', f"'{directory}'"],
        [8, 'text', 'text', _txt]
        ])
    ###################################################################################
    ngp.out(txtpath.replace('.txt', '.ngp'))


def write_polarization_graph(data):
    """
    偏光角-開放端電圧、偏光角-短絡電流のグラフを書き出す
    """
    # 分割
    angles, voltages, currents = data

    # カーブフィッティング
    # formulaの変数を求める
    sv_params = opt.curve_fit(formula, angles, voltages)[0]
    sa_params = opt.curve_fit(formula, angles, currents)[0]

    # 曲線を書く用のx軸データ
    graph_xlim = correct_x(angles[-1])+(ANGLE_ADD*2)
    polar_angles = np.arange(
        angles[0], graph_xlim, step=2)

    # 曲線用のデータを作成
    pred_y = np.vstack([
        [formula(angle, *sv_params) for angle in polar_angles],
        [formula(angle, *sa_params) for angle in polar_angles]
    ])

    # 単位修正
    v_prefixes = get_best_prefix(PREFIX_V)
    i_prefixes = get_best_prefix(PREFIX_I)
    voltages = np.array(voltages)/10**v_prefixes[1]
    currents = np.array(currents)/10**i_prefixes[1]
    pred_y = convert_to_prefix_values(pred_y, v_prefixes[1], i_prefixes[1])
    for i in [0, 2]:
        sv_params[i] = sv_params[i]/10**v_prefixes[1]
        sa_params[i] = sa_params[i]/10**i_prefixes[1]

    # フィッテイング書き出し
    with open(os.path.join(DEST_PATH, 'fitting_result.txt'), 'w') as f:
        f.writelines([
            'Voc fitting\n',
            f'\tAmplitude : {sv_params[0]:>10.6} {v_prefixes[0]}V\n',
            f'\t{"Phase".ljust(9)} : {sv_params[1]:>10.6} deg\n',
            f'\tIntercept : {sv_params[2]:>10.6} {v_prefixes[0]}V\n',
            'Isc fitting\n',
            f'\tAmplitude : {sa_params[0]:>10.6} {i_prefixes[0]}A\n',
            f'\t{"Phase".ljust(9)} : {sa_params[1]:>10.6} deg\n',
            f'\tIntercept : {sa_params[2]:>10.6} {i_prefixes[0]}A\n'
            ])
    # グラフ書き出し
    fig, (graph_l, graph_r) = plt.subplots(ncols=2, figsize=(10,4))

    def plot_voltage():
        """
        偏光角-開放端電圧特性
        """
        graph_l.plot(angles, voltages, linestyle='None', marker='o', color='green')
        graph_l.plot(polar_angles, pred_y[0],
            label=f'Voc={sv_params[0]:.3f}'\
                f'sin(2φ{sv_params[1]*3.1415926535/180:+.1f}){sv_params[2]:+.1f}',
            color='black')
        if IS_DEBUG:
            print('voc:',sv_params)
        y_scales = calc_y_bars(list(graph_l.get_ylim()))
        if y_scales[2] < 1:
            dec_len = len(str(abs(y_scales[2])).split('.')[1])
            y_scales[1] = round(y_scales[1], dec_len)
        ##################################### 変更場所 #####################################
        # 特性グラフの軸の名前
        graph_l.set_title('Voc-φ')
        graph_l.set_xlabel('Angle of light polarization φ (deg)')
        graph_l.set_ylabel(f'Open circuit voltage ({v_prefixes[0]}V)')
        ###################################################################################
        graph_l.set_ylim(y_scales[:2])
        graph_l.set_yticks(
            np.arange(y_scales[0],y_scales[1]+y_scales[2], y_scales[2]))
        graph_l.set_xlim([angles[0], angles[-1]])
        graph_l.set_xticks(
            np.arange(angles[0], graph_xlim, ANGLE_ADD*2))
        graph_l.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
        # ngraph用のデータを書き出す
        update_ngp_mode('v')
        write_ngp_data(
            angles,
            voltages,
            y_scales,
            sv_params,
            os.path.join(DEST_PATH, 'Ngraph data', 'Voc_phi.txt'),
            v_prefixes[0])

    def plot_current():
        """
        偏光角-短絡電流特性
        """
        graph_r.plot(angles, currents, linestyle='None', marker='o', color='blue')
        graph_r.plot(polar_angles, pred_y[1],
            label=f'Isc={sa_params[0]:.3f}'\
                f'sin(2φ{sa_params[1]*math.pi/180:+.1f}){sa_params[2]:+.1f}',
            color='black')
        if IS_DEBUG:
            print('isc:',sa_params)
        y_scales = calc_y_bars(list(graph_r.get_ylim()))
        if y_scales[2] < 1:
            dec_len = len(str(abs(y_scales[2])).split('.')[1])
            y_scales[1] = round(y_scales[1], dec_len)
        ##################################### 変更場所 #####################################
        # 特性グラフの軸の名前
        graph_r.set_title('Isc-φ')
        graph_r.set_xlabel('Angle of light polarization φ (deg)')
        graph_r.set_ylabel(f'Short circuit current ({i_prefixes[0]}A)')
        ###################################################################################
        graph_r.set_ylim(y_scales[:2])
        graph_r.set_yticks(
            np.arange(y_scales[0], y_scales[1]+y_scales[2], y_scales[2]))
        graph_r.set_xlim([angles[0], angles[-1]])
        graph_r.set_xticks(
            np.arange(angles[0], graph_xlim, ANGLE_ADD*2))
        graph_r.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
        update_ngp_mode('i')
        write_ngp_data(
            angles,
            currents,
            y_scales,
            sa_params,
            os.path.join(DEST_PATH, 'Ngraph data', 'Isc_phi.txt'),
            i_prefixes[0])

    plot_voltage()
    plot_current()
    fig.savefig(os.path.join(DEST_PATH, 'Voc_Isc_phi.png'))
    plt.close()


def write_to_excel(data):
    """
    エクセルデータを出力(使用module:openpyxl)
    """
    data = np.array(data).T
    prefix_arr = [PREFIX_V, PREFIX_I, PREFIX_G]
    prefix = []
    for i in range(2, len(data)):
        _prefix, num = get_best_prefix(prefix_arr[i-2])
        prefix.append(_prefix)
        data[i] = data[i]/10**num
    for_avg = data.copy()
    data = data.T.tolist()
    data += [[None, 'Average', np.average(for_avg[2]), np.average(for_avg[3]), np.average(for_avg[4])]]

    # pylint: disable=abstract-class-instantiated
    # pylint: disable=unsubscriptable-object
    output_file = os.path.join(DEST_PATH, f'{os.path.split(INPUT_PATH)[1]}_result.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        ##################################### 変更場所 #####################################
        # excelの内容
        columns = [
            'λ/2 plate angle (deg)',
            'Polarization angle (deg)',
            f'Voltage ({prefix[0]}V)',
            f'Current ({prefix[1]}A)',
            f'Conductance ({prefix[2]}S)'
            ]
        ###################################################################################
        df_data = pd.DataFrame(data, columns=columns)
        df_data.to_excel(writer, sheet_name='result', engine='openpyxl')

        # エクセルのコラム幅を変える
        def change_cellwidth(_key, cells, xcolumns):
            worksheet = writer.book[_key]
            for i, cell in enumerate(cells):
                worksheet.column_dimensions[cell].width = len(columns[i])+1
        change_cellwidth('result', ['B', 'C', 'D', 'E', 'F'], columns)

        for _key in IV_DATA.keys():
            _pv, _pi = IV_DATA[_key]['prefix']
            columns = [
                f'Voltage ({_pv[0]}V)',
                f'Current(real) ({_pi[0]}A)',
                f'Current(fit) ({_pi[0]}A)']
            df_data = pd.DataFrame(IV_DATA[_key]['data'], columns=columns)
            df_data.to_excel(writer, sheet_name=_key, engine='openpyxl')
            change_cellwidth(_key, ['B', 'C', 'D'], columns)


def dat_to_ndarray(filepath):
    """
    ゼロチェックのデータ以降を読み込む
    """
    _array = np.loadtxt(filepath, delimiter=',')
    if _array[0][0] == 0.: # zchk
        _array = np.delete(_array, 0, 0)
    return _array.T


def main():
    """
    データを作成(フォーマットは林本さんの形式)
    """
    global DEST_PATH # pylint: disable=global-statement
    browse() # 参照(INPUT_PATHを更新)
    DEST_PATH = os.path.join(INPUT_PATH, get_curtime()+'_result')
    print('>>> パス : ' + INPUT_PATH)

    error_log = []
    nums = {}
    for _path in iglob(escape_chr(INPUT_PATH)+'/*'):
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
                error_log.append(f"Invalid file name : {_path}\n")
                print(f'\033[38;5;009m>>> [ERROR] ファイルを読み込めません({os.path.split(_path)[1]})'\
                    '。ファイル名を確認してください。\033[38;5;010m')
                continue
        nums[_path] = num
    if len(nums) == 0 and FOR_VEE:
        print(">>>\033[38;5;009m [ERROR] 有効なファイルがありませんでした。\033[38;5;010m")
        sleep(3)
        sys.exit(1)

    if len(nums) == 0:
        print(">>>\033[38;5;009m [ERROR] 有効なファイルがありませんでした。\n\033[38;5;010m>>> ほかのデータを参照しますか? (y/n):", end='')
        return

    nums = sorted(nums.items(), key=lambda x: x[1])
    os.makedirs(DEST_PATH, exist_ok=True)
    os.makedirs(os.path.join(DEST_PATH, 'I-V characteristics'), exist_ok=True)
    os.makedirs(os.path.join(DEST_PATH, 'Ngraph data'), exist_ok=True)
    if len(error_log) > 0:
        output_file = os.path.join(DEST_PATH, 'Error.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(error_log)

    data = []
    polar_angle = 0.
    polarizations = [[],[],[]]
    for _nums in tqdm.tqdm(nums):
        v_oc, i_sc, cond = write_linear_graph(
            dat_to_ndarray(_nums[0]),
            os.path.split(_nums[0])[-1]
        )

        data += [[_nums[1], polar_angle, v_oc, i_sc, cond]]
        polarizations[0] += [polar_angle]
        polarizations[1] += [v_oc]
        polarizations[2] += [i_sc]
        polar_angle += ANGLE_ADD
    write_polarization_graph(polarizations)
    write_to_excel(data)
    write_pathlog()
    if IS_DEBUG:
        shutil.rmtree(DEST_PATH)
        print(sys.argv, '\033[0m')
        sys.exit(0)

    # ファイルを開く
    if os.name == 'nt':
        DEST_PATH = DEST_PATH.replace('/', '\\')
        if __debug__:
            call_cmd(f"explorer {DEST_PATH}")

    if FOR_VEE:
        print(">>> 完了!\033[0m")
        sys.exit(0)
    print(">>> 完了!\n>>> 続行しますか? (y/n):", end='')


def reset():
    """
    グローバル変数のリセット
    """
    # pylint: disable=global-statement
    global DIR_SELECTED
    global DEST_PATH
    global PREFIX_V, PREFIX_I, PREFIX_G
    global IV_DATA
    # pylint: enable=global-statement

    DIR_SELECTED = False
    DEST_PATH = None
    PREFIX_V.clear()
    PREFIX_I.clear()
    PREFIX_G.clear()
    IV_DATA = {}


if __name__=='__main__':
    main()

    while True:
        key = ord(msvcrt.getch())
        sys.stdout.write(f"{chr(key)}")
        if key==3 or chr(key).lower()=='n':
            print('\033[0m')
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
