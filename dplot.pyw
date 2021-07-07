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
        print(">>> 有効なコマンドではありません : " + str(command))
call_cmd("cls")
call_cmd("color 0A")
call_cmd("title .dat plot")

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
print('#'*CMD_TERMINAL_LENGTH)
print("Dat Plot Tool (2021/01/12)")
print(">>> 初回起動は少しロードが遅くなります。")
print(">>> ブラウズ画面やプログレスバーが表示されないときは何かのキーを押してください。")
print('#'*CMD_TERMINAL_LENGTH)
sys.stdout.write("\n")


# pylint: disable=wrong-import-position
from glob import iglob
from glob import escape as escape_chr
from tkinter import Tk
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn import linear_model
# pylint: enable=wrong-import-position


# 下2つは内部で変わるので触らないこと。
# ブラウズ用 すでにフォルダは選択したか?
DIR_SELECTED = False
# ブラウズ後のディレクトリパス
INPUT_PATH = None


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

    if not DIR_SELECTED:
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
            print(">>> キャンセルされました。")
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


def datread_ignorezchk(readfile) -> list:
    """
    ゼロチェックのデータ以降を読み込む
    """
    datacount = 0
    for i, string in enumerate(readfile):
        if string[0] != '#':
            datacount += 1
            if string[:2] == '0,' and datacount < 2:
                return readfile[i+1:]

    return readfile


def write_all_linear_graph(data):
    """
    線形回帰から傾きと切片を求め、グラフを出力
    """
    # 線形モデル

    model, pred_x, pred_y = get_best_model(data)

    coef, intercept = model.coef_[0][0], model.intercept_[0]

    v_oc = intercept/coef

    return v_oc, intercept, coef, data, [pred_x, pred_y]


def set_y_bars(y, yp):
    """
    y軸のスケール幅を設定する
    """

    act_bool = max(max(y), max(yp)) < 0 or min(min(y), min(yp)) > 0
    yma = max(
            abs(max(y)),
            abs(min(y)),
            abs(max(yp)),
            abs(min(yp)))
    ymm = min(
            abs(max(y)),
            abs(min(y)),
            abs(max(yp)),
            abs(min(yp)))
    if act_bool:
        # 基点を0にする
        ref_gap = (yma+ymm)/2
        yma -= ref_gap
        ymm -= ref_gap
    else:
        ref_gap = 0.

    npow = int(math.log10(yma))
    # 軸幅の設定
    if yma/10**npow < 1.25:
        step = 0.2*10**npow
    elif yma/10**npow < 1.25:
        step = 0.5*10**npow
    elif yma/10**npow < 3.5:
        step = 1.0*10**npow
    elif yma/10**npow <= 5.0:
        step = 2.0*10**npow
    else:
        step = 5.0*10**npow

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
        if max(max(y), max(yp)) < 0:
            sgn = -1
        else:
            sgn = 1
        y_min = rangeset(ymm) + fix_gap * sgn
        y_max = rangeset(yma) + fix_gap * sgn
        # さらに補正
        while (
            y_min > min(min(y), min(yp))
            or y_max < max(max(y), max(yp))
        ):
            y_min -= step
            y_max += step
    else:
        y_min = rangeset(min(min(y), min(yp)))
        y_max = rangeset(max(max(yp), max(yp)))
        # 上下の高さを統一する
        y_min = max(abs(y_max), abs(y_min)) * abs(y_min) / y_min
        y_max = max(abs(y_max), abs(y_min)) * abs(y_max) / y_max

    if int(abs((y_max-y_min)/step)) == 2:
        # 軸数が2つの場合は4つにする
        step /= 2

    return y_min, y_max, step


def main():
    """
    データを作成(フォーマットは林本さんの形式)
    """
    browse() # 参照(INPUT_PATHを更新)
    print('>>> パス : ' + INPUT_PATH)

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
            except ValueError:
                print(f'>>> [ERROR] ファイルを読み込めません({os.path.split(_path)[1]})'\
                    '。ファイル名を確認してください。')
                continue
        nums[_path] = num
    if len(nums) == 0:
        print(">>> [ERROR] 有効なファイルがありませんでした。\n>>> ほかのデータを参照しますか? (y/n):", end='')
        return

    nums = sorted(nums.items(), key=lambda x: x[1])

    polar_angle = 0.
    cnt = 0
    mx = 0.
    stp = 0.

    figure = plt.figure(figsize=(10,5))
    cmap = plt.get_cmap("gist_rainbow", len(nums))
    ax = figure.add_subplot(121)
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (A)')
    ax.set_xlim([-200,200])
    ax.hlines([0],-200,200,"white",linestyles='dashed')
    ax.set_facecolor("black")
    for _nums in tqdm.tqdm(nums):
        with open(_nums[0], 'r') as f:
            v_oc, i_sc, cond, data, pred = write_all_linear_graph(
                dat_separate(
                    datread_ignorezchk(f.readlines()))
            )
        ax.plot(pred[0], pred[1],
            linestyle='solid',
            label=f"phi={polar_angle:>6}, Voc={v_oc: .2e}, Isc={i_sc: .2e}, Gphoto={cond: .2e}",
            color=cmap(cnt))
        polar_angle += ANGLE_ADD
        cnt += 1

        _, yb, _stp = set_y_bars(data[1], pred[1])
        if mx < yb:
            mx = yb
            stp = _stp

    ax.set_ylim([-mx, mx])
    ax.vlines([0],-mx, mx,"white",linestyles='dashed')
    ax.set_yticks(np.arange(-mx, mx, stp))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0,))
    write_pathlog()

    figure.savefig(os.path.join(INPUT_PATH, 'all_plot.png'))
    plt.close()

    print(">>> 完了!\n>>> 続行しますか? (y/n):", end='')


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
            DIR_SELECTED = False
            main()
        else:
            print_write = "\n>>> 無効なキーです。「y」か「n」を押してください:"
            if key == 0x1B:
                sys.stdout.write('0')
            sys.stdout.write(print_write)
