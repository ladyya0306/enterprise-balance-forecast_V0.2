"""V3 只读整理补齐：复用既有 compare/曲线规则，不训练或生成。"""
from __future__ import annotations
import csv, json, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from shape_contract_search import compare
from curve_repetition_check import shape as curve_shape, whole_curve_density

ROOT=Path(__file__).resolve().parents[1]
ROWS=json.loads((ROOT/'data/manifests/样本处理清单.json').read_text(encoding='utf-8'))
FINANCE=('股东','银行借款','贷款','借款本金','利息')

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def corr_signature(row):
    folder=ROOT/row['v3_folder']; daily=read(folder/'account_daily_total.csv')[-365:]
    index={r['calendar_date']:i for i,r in enumerate(daily)}; operating=np.zeros(len(daily))
    for n in read(folder/'cash_flow_review_notes.csv'):
        i=index.get(n['booking_datetime'][:10])
        if i is not None and not any(x in n['cash_flow_type_cn'] for x in FINANCE):
            operating[i]+=float(n['amount_cny'])*(1 if n['direction_cn']=='流入' else -1)
    return {'balance':np.asarray([float(x['ending_balance_cny']) for x in daily]),'cash':np.asarray([float(x['inflow_cny'])-float(x['outflow_cny']) for x in daily]),'operating':operating}
def classify(row):
    vals=[float(x['ending_balance_cny']) for x in read(ROOT/row['v3_folder']/'account_daily_total.csv')]
    n=len(vals); q=max(1,n//4); first=sum(vals[:q])/q; last=sum(vals[-q:])/q; lo=min(range(n),key=vals.__getitem__); hi=max(range(n),key=vals.__getitem__); scale=max(np.median(vals),1)
    if lo>n*.25 and lo<n*.75 and last>first*1.05:return '长期U型/后段反弹'
    if hi>n*.25 and hi<n*.75 and last<first*.95:return '冲高回落'
    if last>first*1.08:return '长期上升'
    if last<first*.92:return '长期下降'
    return '长期波动或平稳'
def category(note):
    s=note['cash_flow_type_cn']
    if '工资' in s:return '工资'
    if '税' in s:return '税费'
    if any(x in s for x in ('租赁','房租','场地')):return '租赁场地'
    if any(x in s for x in ('客户','销售','回款')):return '客户回款'
    if any(x in s for x in ('材料','采购','供应')):return '采购材料外协'
    if any(x in s for x in ('设备','固定资产')):return '设备资产'
    if any(x in s for x in ('股东','借款','利息')):return '融资'
    if any(x in s for x in ('维护','服务','合规','支持')):return '经营服务'
    return '未知'
def main():
    sig={r['sample_id']:corr_signature(r) for r in ROWS}; pairs=[]; hints=[]
    ordered=sorted(ROWS,key=lambda x:x['sample_id'])
    for i,left in enumerate(ordered):
        for right in ordered[i+1:]:
            c=compare(sig[left['sample_id']],sig[right['sample_id']])
            # 原 compare 的经营强相关是触发线；仅跨家族才影响隔离。
            if c.get('operating_strong'):
                item={'left':left['sample_id'],'right':right['sample_id'],'left_family':left['family_id'],'right_family':right['family_id'],'balance_correlation':c.get('balance_correlation'),'operating_correlation':c.get('operating_cash_correlation'),'cross_family':left['family_id']!=right['family_id'],'cross_group':left['v3_role']!=right['v3_role']}
                (pairs if item['cross_family'] else hints).append(item)
    # 分类与曲线内重复：复用既有曲线形状门，不因工具不适用写通过。
    result=[]
    for r in ROWS:
        daily=read(ROOT/r['v3_folder']/'account_daily_total.csv'); changes=[float(daily[i]['ending_balance_cny'])-float(daily[i-1]['ending_balance_cny']) for i in range(1,len(daily))]
        months=[]
        for start in range(0,len(changes),30):months.append((str(start),curve_shape(changes[start:start+30])))
        density=whole_curve_density(months)
        result.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'v3_role':r['v3_role'],'measured_shape_v3':classify(r),'curve_repetition_evidence':density,'balance_median_cny':r['balance_median_cny'],'days':r['days']})
    (ROOT/'data/manifests/近复制证据.json').write_text(json.dumps({'method':'复用shape_contract_search.compare的余额/经营收支相关；每户末365自然日','cross_family_strong_pairs':pairs,'same_family_hints':hints},ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'data/manifests/实测走势与曲线重复.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    # 仅学习组出现过的类别组成字典；其他组未见类别归未知。
    learn=set(); raw=[]
    for r in ROWS:
        for n in read(ROOT/r['v3_folder']/'cash_flow_review_notes.csv'):
            c=category(n)
            if r['v3_role']=='学习' and c!='未知':learn.add(c)
            raw.append((r,n,c))
    out=ROOT/'data/manifests/历史用途输入.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','family_id','v3_role','transaction_id','booking_date','direction','amount_cny','usage_class','usage_assumption','source_note_type'])
        for r,n,c in raw:w.writerow([r['sample_id'],r['family_id'],r['v3_role'],n['transaction_id'],n['booking_datetime'][:10],n['direction_cn'],n['amount_cny'],c if c in learn else '未知','模拟入账时可获得；非真实银行标签时间',n['cash_flow_type_cn']])
    with (ROOT/'data/manifests/样本分组名单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','family_id','开发资料分组','日余额CSV','逐笔流水CSV','备注CSV','用途输入CSV']);
        for r in ROWS:w.writerow([r['sample_id'],r['family_id'],r['v3_role'],r['v3_folder']+'/account_daily_total.csv',r['v3_folder']+'/transactions_total.csv',r['v3_folder']+'/cash_flow_review_notes.csv','data/manifests/历史用途输入.csv'])
    with (ROOT/'data/manifests/图册CSV对应.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','图页','图中位置','日余额CSV']);
        for i,r in enumerate(ROWS):w.writerow([r['sample_id'],i//8+1,i%8+1,r['v3_folder']+'/account_daily_total.csv'])
    print(json.dumps({'strong_cross_family_pairs':len(pairs),'same_family_hints':len(hints),'shapes':dict(Counter(x['measured_shape_v3'] for x in result)),'learning_usage_dictionary':sorted(learn)},ensure_ascii=False))
if __name__=='__main__':main()
