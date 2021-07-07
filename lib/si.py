"""
SI接頭辞
"""

import math
import sys
import numpy as np

import sys

_VER = sys.version_info
if _VER[1] > 7 and _VER[2] > 1:
    from statistics import mode
    _PROCESS = False
else:
    from scipy.stats import mode
    _PROCESS = True

del _VER


SI_LIST = {
    '24' : ('Y', 24),
    '21' : ('Z', 21),
    '18' : ('E', 18),
    '15' : ('P', 15),
    '12' : ('T', 12),
    '9' : ('G', 9),
    '6' : ('M', 6),
    '3' : ('k', 3),
    '0' : ('', 0),
    '-3' : ('m', -3),
    '-6' : ('μ', -6),
    '-9' : ('n', -9),
    '-12' : ('p', -12),
    '-15' : ('f', -15),
    '-18' : ('a', -18),
    '-21' : ('z', -21),
    '-24' : ('y', -24)
}


def _define_si_list():
    """上の値以外について定義"""
    global SI_LIST

    update_dicts = {}
    for i in range(-24, 25, 3):
        update_dicts[str(i+1)] = SI_LIST[str(i)]
        update_dicts[str(i+2)] = SI_LIST[str(i)]

    SI_LIST.update(**update_dicts)


_define_si_list()


def _list_prefix(any_array):
    prefix_array = []
    for _value in any_array:
        try:
            x = round(math.log10(abs(_value)))
        except (ValueError, TypeError):
            continue
        else:
            try:
                _ = SI_LIST[str(x)]
            except KeyError:
                continue
            else:
                prefix_array.append(x)
    if len(prefix_array) == 0:
        return SI_LIST['0']
    return get_best_prefix(prefix_array)


def get_prefix(n) -> tuple:
    """接頭辞とexpを取得"""
    if isinstance(n, (list, tuple, np.ndarray)):
        if isinstance(n, np.ndarray):
            if n.ndim >1:
                print(f'>>>\033[38;5;009m [ERROR] : ndarrayの次元は1である必要があります(引数のndarrayの次元 : {n.ndim})\033[0m')
                sys.exit(1) 
        return _list_prefix(n)
    else:
        try:
            x = round(math.log10(abs(n)))
        except (ValueError, TypeError):
            print('>>>\033[38;5;009m [ERROR] : 無効な型です : ', type(n), end="\033[0m\n")
            sys.exit(1)
        else:
            try:
                ret = SI_LIST[str(x)]
            except KeyError:
                print(f'>>>\033[38;5;009m [ERROR] : 無効な値を取得しました : prefix = {x}\033[0m')
                sys.exit(1)
        return ret


def get_best_prefix(n_list) -> tuple:
    """リストから最も多い要素の接頭辞を返す"""
    _mode = mode(n_list)
    if _PROCESS:
        _mode = _mode[0][0]

    return SI_LIST[str(_mode)]