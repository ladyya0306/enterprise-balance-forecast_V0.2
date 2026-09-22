from pathlib import Path
import csv
root=Path(__file__).resolve().parents[1]; source=root/'data/manifests/历史用途输入.csv'
with source.open(encoding='utf-8-sig',newline='') as f:
 r=csv.reader(f); header=next(r); writers=[]; handles=[]
 for n in (1,2):
  h=(root/f'data/manifests/历史用途输入_第{n}部分.csv').open('w',encoding='utf-8-sig',newline='');handles.append(h);w=csv.writer(h);w.writerow(header);writers.append(w)
 for i,row in enumerate(r):writers[i%2].writerow(row)
for h in handles:h.close()
