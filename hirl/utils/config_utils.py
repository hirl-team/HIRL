from typing import List, Tuple
from addict import Dict as adict

def is_number_or_bool_or_none(x):
    try:
        float(x)
        return True
    except ValueError:
        return x in ['True', 'False', 'None']


def add_quotation_to_string(s: str,
                            split_chars: List[str] = None) -> str:
    if split_chars is None:
        split_chars = ['[', ']', '{', '}', ',', ' ']
        if '{' in s and '}' in s:
            split_chars.append(':')
    s_mark, marker = s, chr(1)
    for split_char in split_chars:
        s_mark = s_mark.replace(split_char, marker)

    s_quoted = ''
    for value in s_mark.split(marker):
        if len(value) == 0:
            continue
        st = s.find(value)
        if is_number_or_bool_or_none(value):
            s_quoted += s[:st] + value
        elif value.startswith("'") and value.endswith("'") or value.startswith('"') and value.endswith('"'):
            s_quoted += s[:st] + value
        else:
            s_quoted += s[:st] + '"' + value + '"'
        s = s[st + len(value):]

    return s_quoted + s

def update_config(cfg: adict,
                  cfg_argv: List[str],
                  delimiter: str = '=') -> None:
    def resolve_cfg_with_legality_check(keys: List[str]) -> Tuple[adict, str]:
        obj, obj_repr = cfg, 'cfg'
        for idx, sub_key in enumerate(keys):
            if not isinstance(obj, adict) or sub_key not in obj:
                raise ValueError(f'Undefined attribute "{sub_key}" detected for "{obj_repr}"')
            if idx < len(keys) - 1:
                obj = obj.get(sub_key)
                obj_repr += f'.{sub_key}'
        return obj, sub_key

    if len(cfg_argv) == 0:
        return
    _2adict(cfg)

    for str_argv in cfg_argv:
        item = str_argv.split(delimiter, 1)
        assert len(item) == 2, "Error argv (must be key=value): " + str_argv
        key, value = item
        obj, leaf = resolve_cfg_with_legality_check(key.split('.'))
        obj[leaf] = eval(add_quotation_to_string(value))

def _2adict(cfg):
    for key, value in cfg.items():
        if key != 'model_bak':
            cfg[key] = adict(tmp=value)['tmp']