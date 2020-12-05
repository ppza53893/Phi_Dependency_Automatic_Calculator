"""
ngpを書き出すためのpyファイル
"""
import json
import os
from distutils.util import strtobool


# pylint: disable=invalid-name
class NgraphParser:
    """
    ngpからjson形式へ変換。

    使い方:
    ```python
    from ngraph import NgraphParser
    ngp = NgraphParser()
    ngp.convert('sample.ngp', 'sample.json')
    ```
    """
    def __init__(self, encoding=None):
        self.encoding = 'shift_jis' if not encoding else encoding
        self.groups = []

    def convert(self, fpath, dest):
        """
        example:
        ```python
        from ngraph import NgraphParser
        ngp = NgraphParser()
        ngp.convert('sample.ngp', 'sample.json')
        ```
        """
        if not os.path.exists(fpath):
            raise FileNotFoundError("Couldn't open file : {}".format(fpath))
        with open(fpath, encoding=self.encoding) as fp:
            self.read(fp)
        with open(dest, 'w') as wp:
            json.dump({'data':self.groups}, wp, indent=2)

    def read(self, fp):
        """
        ngpを解析。
        """
        _fp = fp.readlines()
        comments = []
        for i, line in enumerate(_fp):
            if '#' in line:
                if line.find('#') == 0:
                    comments.append(line[0:].strip())
                else:
                    continue
            elif line == '\n':
                continue
            elif 'new' in line and line.find('new') == 0:
                name = line.rstrip()[line.find(' ')+1:]
                self.read_opt(name, _fp[i+1:])
            elif 'grouping' in line:
                self.groups.append({
                    'grouping':[self.str2num(j) for j in line.rstrip().split(' ')[1:]]
                })

    @classmethod
    def str2bool(cls, string) -> bool:
        '''
        convert str to boolean
        '''
        return bool(strtobool(string))

    @classmethod
    def str2num(cls, string):
        '''
        convert str to int or float.
        '''
        try:
            p = int(string)
        except ValueError:
            p = float(string)
        return p

    def read_opt(self, group_name, lines):
        """
        ngpのパラメータをpython形式に変換し、dict型にする。
        """
        options = {}
        for _, line in enumerate(lines):
            if line =='\n' or '::' not in line:
                break

            line = line.lstrip()
            name = line[line.find('::')+2:line.find('=')]
            p = line[line.find('=')+1:-1]
            if p in ['true', 'false']:
                param = NgraphParser.str2bool(p)
            elif "'" in p:
                param = p
            elif len(p) == 0:
                param = None
            else:
                try:
                    param = float(p)
                except ValueError:
                    param = p
                else:
                    param = NgraphParser.str2num(p)
            options[name] = param
        self.groups.append({group_name:options})


class NgraphWriter:
    """
    Ngraphデータの更新をしたり書き出したりする。
    """
    def __init__(self, mode='v', filepath=None):
        self.data = None
        if filepath is None:
            if mode=='v':
                _json = 'base_voc.json'
            elif mode=='i':
                _json = 'base_isc.json'
            else:
                raise ValueError('Invalid mode : '+mode)
            self.current_path = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                'json', _json)
        else:
            self.current_path = filepath
        self.read()
        self.writetext = []

    def read(self):
        """
        データを読み込む。初めにされるのでこの関数は指定しなくてもよい。
        """
        with open(self.current_path, 'r') as f:
            self.data = f.read()
            self.data = json.loads(self.data)['data']

    def write(self, parameters):
        """
        データを更新する。
        """
        try:
            _x = parameters[0][0]
        except (IndexError, TypeError):
            parameters = [parameters]
        for p in parameters:
            self.data[p[0]][p[1]][p[2]] = p[3]

    def ngp_append(self, obj='', hastab=False):
        """
        書き出し用リストに要素を追加する。
        """
        append_str = ''
        if hastab:
            append_str += '\t'
        append_str += str(obj) + '\n'
        self.writetext.append(append_str)

    def out(self, dest):
        """
        ngpファイルを書き出す。
        """
        for _data in self.data:
            classname, cfg = list(_data.items())[0]
            if classname == 'grouping':
                self.ngp_append(
                    'axis::'+classname + ' ' + str(cfg)[1:-1].replace(',',''))
            else:
                self.ngp_append('new '+ classname)
                for conf, param in cfg.items():
                    _cls = classname.split(' ')[0] if ' ' in classname\
                        else classname
                    if isinstance(param, bool):
                        param = str(param).lower()
                    if param is None:
                        param = ''
                    self.ngp_append(_cls + '::' +\
                        conf+'='+str(param), True)
            if classname != 'fit':
                self.ngp_append()
        with open(dest, 'w') as w:
            w.writelines(self.writetext)
