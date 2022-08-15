# coding = utf-8
import os, re, sys, random, urllib.parse, json
from collections import defaultdict
import requests


def write_line(file_path, sentences, sep='\t'):
    with open(file_path, 'w+') as f:
        f.write(sep.join([str(x) for x in sentences]) + '\n')


def regex_match(patt, sr):
    match = re.search(patt, sr, re.DOTALL | re.MULTILINE)
    return match.group(0) if match else ''


def get_page(url, cookie='', proxy='', timeout=5):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
        if cookie != '': headers['cookie'] = cookie
        if proxy != '':
            proxies = {'http': proxy, 'https': proxy}
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
        else:
            resp = requests.get(url, headers=headers, timeout=timeout)
        content = resp.content
        try:
            import chardet
            charset = chardet.detect(content).get('encoding', 'utf-8')
            if charset.lower().startswith('gb'): charset = 'gbk'
            content = content.decode(charset, errors='replace')
        except:
            headc = content[:min([3000, len(content)])].decode(errors='ignore')
            charset = regex_match('charset="?([-a-zA-Z0-9]+)', headc)
            if charset == '': charset = 'utf-8'
            content = content.decode(charset, errors='replace')
    except Exception as e:
        print(e)
        content = ''
    return content


def get_json(url, cookie='', proxy='', timeout=5.0):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
        if cookie != '': headers['cookie'] = cookie
        if proxy != '':
            proxies = {'http': proxy, 'https': proxy}
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
        else:
            resp = requests.get(url, headers=headers, timeout=timeout)
        return resp.json()
    except Exception as e:
        print(e)
        content = {}
    return content


def find_all_hrefs(url, content=None, regex=''):
    ret = set()
    if content is None:
        content = get_page(url)
    patt = re.compile('href="?([a-zA-Z0-9-_:/.%]+)')
    for xx in re.findall(patt, content):
        ret.add(urllib.parse.urljoin(url, xx))
    if regex != '': ret = (x for x in ret if re.match(regex, x))
    return list(ret)


def translate(txt):
    post_data = {'from': 'en', 'to': 'zh', 'transtype': 'realtime', 'query': txt}
    url = "http://fanyi.baidu.com/v2transapi"
    try:
        resp = requests.post(url, data=post_data, headers={'Referer': 'http://fanyi.baidu.com/'})
        ret = resp.json()
        print(ret)
        ret = ret['trans_result']['data'][0]['dst']
    except Exception as e:
        print(e)
        ret = ''
    return ret


def is_chinese_text(z):
    return re.search('^[\u4e00-\u9fa5]+$', z) is not None


def freq_dict_to_list(dt):
    return sorted(dt.items(), key=lambda d: d[-1], reverse=True)


def select_rows_by_column(fn, ofn, st, num=0):
    with open(fn, encoding="utf-8") as fin:
        with open(ofn, "w", encoding="utf-8") as fout:
            for line in (ll for ll in fin.read().split('\n') if ll != ""):
                if line.split('\t')[num] in st:
                    fout.write(line + '\n')


def merge_files(dir, obj_file, regex=".*"):
    with open(obj_file, "w", encoding="utf-8") as f_out:
        for file in os.listdir(dir):
            if re.match(regex, file):
                with open(os.path.join(dir, file), encoding="utf-8") as f_in:
                    f_out.write(f_in.read())


def join_files(file_x, file_y, file_out):
    with open(file_x, encoding="utf-8") as f_in:
        lx = [vv for vv in f_in.read().split('\n') if vv != ""]
    with open(file_y, encoding="utf-8") as f_in:
        ly = [vv for vv in f_in.read().split('\n') if vv != ""]
    with open(file_out, "w", encoding="utf-8") as f_out:
        for i in range(min(len(lx), len(ly))):
            f_out.write(lx[i] + "\t" + ly[i] + "\n")


def remove_duplicated_rows(file, obj_file='*'):
    st = set()
    if obj_file == '*':
        obj_file = file
    with open(file, encoding="utf-8") as f_in:
        for line in f_in.read().split('\n'):
            if line == "":
                continue
            st.add(line)
    with open(obj_file, "w", encoding="utf-8") as f_out:
        for line in st:
            f_out.write(line + '\n')


def load_csv(file, sep='\t'):
    ret = []
    with open(file, encoding='utf-8') as f_in:
        for line in f_in:
            lln = line.rstrip('\r\n').split(sep)
            if len(lln) == 0 or len(lln[0]) == 0:
                continue
            ret.append(lln)
    return ret


def load_csv_generator(file, sep='\t'):
    with open(file, encoding='utf-8') as f_in:
        for line in f_in:
            lln = line.rstrip('\r\n').split(sep)
            if len(lln) == 0 or len(lln[0]) == 0:
                continue
            yield lln


def save_csv(csv, file, sep='\t'):
    with open(file, 'w', encoding='utf-8') as f_out:
        for x in csv:
            f_out.write(sep.join(x) + "\n")


def split_table(file, limit=3):
    rst = set()
    with open(file, encoding='utf-8') as f_in:
        for line in f_in:
            lln = line.rstrip('\r\n').split('\t')
            rst.add(len(lln))
    if len(rst) > limit:
        print('%d tables, exceed limit %d' % (len(rst), limit))
        return
    for ii in rst:
        print('%d columns' % ii)
        with open(file.replace('.txt', '') + '.split.%d.txt' % ii, 'w', encoding='utf-8') as fout:
            with open(file, encoding='utf-8') as fin:
                for line in fin:
                    lln = line.rstrip('\r\n').split('\t')
                    if len(lln) == ii:
                        fout.write(line)


def load_set(file):
    with open(file, encoding="utf-8") as f_in:
        line_set = set(line for line in f_in.read().split('\n') if line != "")
    return line_set


def load_jsons(file):
    with open(file, encoding='utf-8') as f_in:
        line_list = [json.loads(line) for line in f_in.read().split("\n") if line != ""]
    return line_list


def load_list(file):
    with open(file, encoding="utf-8") as f_in:
        line_list = list(line for line in f_in.read().split('\n') if line != "")
    return line_list


def load_list_generator(file):
    with open(file, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                yield line


def load_dict(file, func=str, sep='\t'):
    dict = {}
    with open(file, encoding="utf-8") as fin:
        for line_split in (line.split(sep, 1) for line in fin.read().split('\n') if line != ""):
            dict[line_split[0]] = func(line_split[1])
    return dict


def save_dict(dict, file, sep='\t', output_zero=True):
    with open(file, "w", encoding="utf-8") as fout:
        for k in dict.keys():
            if output_zero or dict[k] != 0:
                fout.write(str(k) + sep + str(dict[k]) + "\n")


def save_list(ls, file_out):
    with open(file_out, "w", encoding="utf-8") as f_out:
        for k in ls:
            f_out.write(str(k) + "\n")


def save_2d_list(ls, file_out):
    with open(file_out, 'w+', encoding='utf-8') as f_out:
        for row in ls:
            for item in row:
                f_out.write(str(item) + "\n")
            f_out.write("\n")


def list_dir_files(dir, filter=None):
    if filter is None:
        return [os.path.join(dir, x) for x in os.listdir(dir)]
    return [os.path.join(dir, x) for x in os.listdir(dir) if filter(x)]


def process_dir(dir, func, param):
    for file in os.listdir(dir):
        print(file)
        func(os.path.join(dir, file), param)


def get_lines(file):
    with open(file, encoding="utf-8", errors='ignore') as f_in:
        lines = list(map(str.strip, f_in.readlines()))
    return lines


def sort_rows(file, obj_file, cid, dtype=int, reverse=True):
    lines = load_csv(file)
    rows = []
    for dv in lines:
        if len(dv) <= cid:
            continue
        rows.append((dtype(dv[cid]), dv))
    with open(obj_file, "w", encoding="utf-8") as f_out:
        for dd in sorted(rows, reverse=reverse):
            f_out.write('\t'.join(dd[1]) + '\n')


def sample_rows(file, obj_file, num):
    zz = list(open(file, encoding='utf-8'))
    num = min([num, len(zz)])
    zz = random.sample(zz, num)
    with open(obj_file, 'w', encoding='utf-8') as f_out:
        for xx in zz:
            f_out.write(xx)


def set_product(file_x, file_y, obj_file):
    l1, l2 = get_lines(file_x), get_lines(file_y)
    with open(obj_file, 'w', encoding='utf-8') as f_out:
        for z1 in l1:
            for z2 in l2:
                f_out.write(z1 + z2 + '\n')


class TokenList:
    def __init__(self, file, low_freq=2, source=None, func=None, save_low_freq=2, special_marks=[]):
        if not os.path.exists(file):
            token_dict = defaultdict(int)
            for i, xx in enumerate(special_marks):
                token_dict[xx] = 100000000 - i
            for xx in source:
                for token in func(xx):
                    token_dict[token] += 1
            tokens = freq_dict_to_list(token_dict)
            tokens = [(x[0], str(x[1])) for x in tokens if x[1] >= save_low_freq]
            save_csv(tokens, file)
        self.id2token = ['<PAD>', '<UNK>'] + \
                        [x for x, y in load_csv(file) if float(y) >= low_freq]
        self.token2id = {v: k for k, v in enumerate(self.id2token)}

    def get_id(self, token):
        return self.token2id.get(token, 1)

    def get_token(self, ii):
        return self.id2token[ii]

    def get_size(self):
        return len(self.id2token)


def cal_f1(correct, output, golden):
    p = correct / output
    r = correct / golden
    f1 = 2 * p * r / (p + r)
    result = 'Precision: %.4f %d/%d, Recall: %.4f %d/%d, F1: %.4f' % (p, correct, output, r, correct, golden, f1)
    return result

def sql(cmd=''):
    if cmd == '':
        cmd = input("> ")
    cts = [x for x in cmd.strip().lower()]
    instr = False
    for i in range(len(cts)):
        if cts[i] == '"' and cts[i - 1] != '\\': instr = not instr
        if cts[i] == ' ' and instr: cts[i] = "&nbsp;"
    cmds = "".join(cts).split(' ')
    keyw = {'select', 'from', 'to', 'where'}
    ct, kn = {}, ''
    for xx in cmds:
        if xx in keyw:
            kn = xx
        else:
            ct[kn] = ct.get(kn, "") + " " + xx

    for xx in ct.keys():
        ct[xx] = ct[xx].replace("&nbsp;", " ").strip()

    if ct.get('where', "") == "": ct['where'] = 'True'

    if os.path.isdir(ct['from']):
        fl = [os.path.join(ct['from'], x) for x in os.listdir(ct['from'])]
    else:
        fl = ct['from'].split('+')

    if ct.get('to', "") == "": ct['to'] = 'temp.txt'

    for xx in ct.keys():
        print(xx + " : " + ct[xx])

    total = 0
    with open(ct['to'], 'w', encoding='utf-8') as fout:
        for fn in fl:
            print('selecting ' + fn)
            for xx in open(fn, encoding='utf-8'):
                x = xx.rstrip('\r\n').split('\t')
                if eval(ct['where']):
                    if ct['select'] == '*':
                        res = "\t".join(x) + '\n'
                    else:
                        res = "\t".join(eval('[' + ct['select'] + ']')) + '\n'
                    fout.write(res)
                    total += 1

    print('completed, ' + str(total) + " records")


def cmd():
    while True:
        cmd = input("> ")
        sql(cmd)


def check_split(file):
    file = file.replace('pretrain', '').replace('eval', '').replace('constrain', '')
    split = ['train', 'val', 'test']
    for s in split:
        if s in file:
            return s
    return 'infer'


def cyclic_merge(a, b):
    a_len, b_len = len(a), len(b)
    min_len = min(a_len, b_len)
    for i in range(min_len):
        yield a[i]
        yield b[i]

    if a_len > b_len:
        for i in range(min_len, a_len):
            yield a[i]
    else:
        for i in range(min_len, b_len):
            yield b[i]


if __name__ == '__main__':
    print(translate('How are you'))