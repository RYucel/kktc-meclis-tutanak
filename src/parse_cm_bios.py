"""Derive per-term membership evidence for pre-1985 assemblies from cm.gov.ct.tr former-MP biographies.

Input : a JSON list of {surname, given, region, doc_url, text} (cm.gov.ct.tr/GecmisMiletvekilleri, with
        .doc/.docx bios decoded), e.g. cm_gecmis_milletvekilleri.json.
Output: data/cm_bios_terms.json - one record per person with 'terms' {term_id: {evidence, party, constituency}}.
        Term ids follow optimize_for_wasm: TCM, KC, KTY:2, KTFD_KM:1, KTFD:1, KTFD:2, KKTC_KM:1.
Usage : python src/parse_cm_bios.py cm_gecmis_milletvekilleri.json [data/cm_bios_terms.json]
"""
import sys
import json, re, unicodedata
from collections import defaultdict

M = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'cm_gecmis_milletvekilleri.json', encoding='utf-8'))

def tr_lower(s): return s.replace('İ','i').replace('I','ı').lower()
def tr_title(w):
    w = tr_lower(w)
    return (w[0].replace('i','İ').replace('ı','I').upper() + w[1:]) if w else w

PARTY = {
    'ulusal birlik partisi':'UBP','cumhuriyetçi türk partisi':'CTP','toplumcu kurtuluş partisi':'TKP',
    'demokratik halk partisi':'DHP','halkçı parti':'HP','türk birliği partisi':'TBP','demokrat parti':'DP',
    'yeni doğuş partisi':'YDP','toplumcu demokrasi partisi':'TDP','milli birlik partisi':'MBP',
    'kıbrıs türk kurtuluş partisi':'KTKP','sosyal demokrat parti':'SDP','ulusal dirliş partisi':'UDP',
}
REGION = {'lefkoşa':'Lefkoşa','lefkosa':'Lefkoşa','mağusa':'Gazimağusa','gazimağusa':'Gazimağusa','gazi mağusa':'Gazimağusa',
          'girne':'Girne','güzelyurt':'Güzelyurt','iskele':'İskele','lefke':'Lefke','baf':'Baf','limasol':'Limasol','larnaka':'Larnaka'}

def sentences(text):
    t = re.sub(r'(?<![.!?:;])\n(?=\S)', ' ', text)          # unwrap broken lines
    t = re.sub(r'(\w)\s*\n\s*(\w)', r'\1 \2', t)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'(?<=[a-zçğıöşü)’\'])([.!?])(?=[A-ZÇĞİÖŞÜ0-9])', r'\1 ', t)   # "oldu.1970" -> "oldu. 1970"
    return [s.strip() for s in re.split(r'(?<=[a-zçğıöşü)’\'0-9][.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])', t) if s.strip()]

ELECTED = re.compile(r'seçil|seçim(?:ler)?(?:i|in)?de.*(?:milletvekil|üye)|milletvekili olarak|üyeliğ|üyelik yaptı|üyesi (?:olarak|oldu)|görev (?:aldı|yaptı)|meclis.{0,20}girdi', re.I)
NEG = re.compile(r'kazanamad|seçilemedi|aday ol(?:du|arak katıldı)(?!.*seçil)', re.I)

def _fold(x):
    return unicodedata.normalize('NFKD', tr_lower(x).replace('ı', 'i')).encode('ascii', 'ignore').decode()

def display_name(m):
    """Name as printed in the bio when possible (keeps ğ/ş/ı), else rebuilt from the list columns."""
    text = m['text']
    raw = (m['given'] + ' ' + m['surname'])
    aliases = re.findall(r'\(([^)]+)\)', raw + ' ' + text.split('\n', 1)[0])
    raw = re.sub(r'\([^)]*\)', ' ', raw)
    toks = [t for t in re.split(r'[\s.]+', raw) if t and not re.fullmatch(r'(?:dr|dt|av|prof|doç)', t, re.I)]
    # order: given names then surname
    sur = [t for t in m['surname'].split() if not re.fullmatch(r'(?:DR|DT|AV)\.?', t, re.I) and '(' not in t]
    sur = [re.sub(r'\(.*', '', t) for t in sur][:1]
    given = [t for t in toks if _fold(t) not in {_fold(x) for x in sur}]
    ordered = given + sur
    words = re.findall(r'[A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+', text[:4000])
    spell = {}
    for w in words:
        fw = _fold(w)
        if fw not in spell and any(c.islower() for c in w[1:]):
            spell[fw] = w
    out = []
    for t in ordered:
        w = spell.get(_fold(t))
        out.append(tr_title(w) if w else tr_title(t))
    return ' '.join(out), [a.strip() for a in aliases if a.strip()]

ASSEMBLY_KW = {
    'TCM': r'cemaat meclis', 'KC': r'temsilciler meclis|kıbrıs cumhuriyeti(?: 1\.)? meclis',
    'KM': r'kurucu meclis', 'FED': r'federe (?:devlet(?:i)? )?meclis|kıbrıs türk federe meclis',
}
WIN_BEFORE, WIN_AFTER = 110, 130

def nearest(win, pos, keys):
    best, bd = None, 10**9
    for k in keys:
        for m in re.finditer(ASSEMBLY_KW[k], win):
            d = abs(m.start() - pos)
            if d < bd:
                best, bd = k, d
    return best

def classify(s):
    sl = tr_lower(s)
    if NEG.search(s):
        return []
    out = []
    for m in re.finditer(r'\b(19[5-9]\d|20[0-2]\d)\b', s):
        y = int(m.group(1))
        if y not in (1960, 1961, 1963, 1970, 1975, 1976, 1981, 1983):
            continue
        pre = s[max(0, m.start() - 8):m.start()]
        if re.search(r'\d{4}\s*[-–]\s*$', pre):          # end of a "1976-1981" range
            continue
        a, b = max(0, m.start() - WIN_BEFORE), m.end() + WIN_AFTER
        win = sl[a:b]
        pos = m.start() - a
        after = sl[m.end():m.end() + 45]
        if re.search(r'^\W{0,3}(?:’|\')?[aeıiouü]?y?[ae]? kadar|sona er|son bul|kadar sürdür|kadar devam|tarihine kadar|ayrıl|bitti|istifa', after):
            continue                                       # an end date, not an election/start date
        if not re.search(r'seçil|seçim|üye|görev|milletvekil|meclis.{0,15}girdi|yemin', win):
            continue
        if re.search(r'devlet başkan|cumhurbaşkan|belediye|müsteşar|murakıp|müfettiş|yargıç|mahkeme|dairesi|temsilcisi olarak avrupa|atan(?:dı|an|arak)|genel sekreter|müdür', win[max(0, pos - 60):pos + 60]):
            continue
        if y in (1960, 1961, 1963):
            k = nearest(win, pos, ['TCM', 'KC'])
            if k == 'TCM' and y in (1960, 1961): out.append('TCM')
            elif k == 'KC': out.append('KC')
        elif y == 1970:
            if nearest(win, pos, ['TCM', 'KC']): out.append('KTY:2')
        elif y == 1975:
            if re.search(ASSEMBLY_KW['KM'], win) or re.search(r'(?:13|22|24) şubat 1975', win) and 'milletvekil' in win:
                out.append('KTFD_KM:1')
        elif y in (1976, 1981):
            if re.search(r'seçim|seçil', win) and re.search(r'milletvekil|federe meclis|meclis.{0,15}girdi', win):
                if nearest(win, pos, ['KM']) and abs(win.find('kurucu meclis') - pos) < 25:
                    continue
                out.append('KTFD:1' if y == 1976 else 'KTFD:2')
        elif y == 1983:
            if re.search(ASSEMBLY_KW['KM'], win): out.append('KKTC_KM:1')
    return list(dict.fromkeys(out))

def details(s):
    sl = tr_lower(s)
    party = next((v for k, v in PARTY.items() if k in sl), None)
    region = None
    m = re.search(r'(lefkoşa|lefkosa|gazi ?mağusa|mağusa|girne|güzelyurt|iskele|lefke|baf|limasol|larnaka)(?:’|\')?(?:dan|den| bölgesi| milletvekili|’dan|’den)', sl)
    if m: region = REGION.get(m.group(1).replace('gazi mağusa', 'gazimağusa'), None)
    return party, region

people = []
for m in M:
    name, paren_aliases = display_name(m)
    hits = defaultdict(list)
    for s in sentences(m['text']):
        for tid in classify(s):
            hits[tid].append(s)
    alias = None
    am = re.search(r'bak[ıi]n[ıi]z\s+([^\n]{3,60})', m['text'], re.I)
    if am:
        alias = am.group(1).strip()
    people.append({'name': name, 'surname_raw': m['surname'], 'given_raw': m['given'], 'birth_region': m['region'],
                   'doc_url': m['doc_url'], 'alias_of': alias, 'aliases': paren_aliases,
                   'terms': {k: {'evidence': v[0][:300], 'party': details(v[0])[0], 'constituency': details(v[0])[1]} for k, v in hits.items()}})
seen = {}
for p in people:
    if p['name'] in seen:
        seen[p['name']]['terms'].update({k: v for k, v in p['terms'].items() if k not in seen[p['name']]['terms']})
    else:
        seen[p['name']] = p
people = list(seen.values())
json.dump(people, open(sys.argv[2] if len(sys.argv) > 2 else 'data/cm_bios_terms.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
from collections import Counter
c = Counter(t for p in people for t in p['terms'])
print(c)
print('with any pre-1985 term:', sum(1 for p in people if p['terms']))
print('aliases:', [(p['name'], p['alias_of']) for p in people if p['alias_of']])
print('names sample:', [p['name'] for p in people[:25]])
