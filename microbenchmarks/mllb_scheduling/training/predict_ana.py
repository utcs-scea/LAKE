import pandas as pd

def predict_ana(df):
    total = len(df)
    stats = {}
    stats['positives'] = df.can_migrate.eq(1).sum()
    stats['false-negative (1,0)'] = (df.can_migrate.eq(1) & df.prediction.eq(0)).sum()
    stats['false-positive (0,1)'] = (df.can_migrate.eq(0) & df.prediction.eq(1)).sum()
    stats['true-positive (1,1)'] = (df.can_migrate.eq(1) & df.prediction.eq(1)).sum()
    stats['true-negative (0,0)'] = (df.can_migrate.eq(0) & df.prediction.eq(0)).sum()
    stats['total'] = len(df)
    for item in stats:
        print(item, stats[item], '{:.4f}'.format(stats[item] / total))

#  pd.set_option('display.max_rows', df.shape[0]+1)

#  #  print(sel.loc[:, ['ts', 'pid', 'src_cpu', 'dst_cpu', 'delta_faults', 'can_migrate']])
#  print(sel)
#  print(len(sel))
#  sel = df[df['delta_faults'].ne(0) & df['prefer_src'].eq(0) & df['prefer_dst'].eq(0)]
#  sel = df[df['prefer_src'].eq(0) & df['prefer_dst'].eq(0)]
#  print(sel[['prefer_src', 'prefer_dst', 'delta', 'delta_faults', 'can_migrate', 'src_numa_len']])
