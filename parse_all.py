import pandas as pd
from powersimdata import Grid

texas = Grid(["Texas"])


def parse_output(dt):
    # Read output and sort by appearance-order
    lmps = pd.read_csv('outputs/lmps_{}.csv'.format(dt.strftime('%Y-%m-%d_%H')), index_col=0).sort_index().dropna(how='all')
    gens = pd.read_csv('outputs/gens_{}.csv'.format(dt.strftime('%Y-%m-%d_%H')), index_col=0).sort_index().dropna(how='all')
    branches = pd.read_csv('outputs/branches_{}.csv'.format(dt.strftime('%Y-%m-%d_%H'), index_col=0)).sort_index().dropna(how='all')
    gens.index     = texas.plant.index
    branches.index = texas.branch.index
    return lmps, gens, branches


lmps_all = {}
gens_all = {}
branches_all = {}
for dt in pd.date_range('2016-1-1 0:00', '2016-12-31 23:00', freq='H'):
    print(dt)
    lmps, gens, branches = parse_output(dt)
    lmps_all[dt] = lmps
    gens_all[dt] = gens
    branches_all[dt] = branches


pg = pd.concat(gens_all)['Column2'].unstack() * 100
lmp = (pd.concat(lmps_all)['Column3']/-100).unstack().round(2)
bdf = pd.concat(branches_all)
bdf = bdf[bdf[['Column3','Column4']].sum(1)>0.1].copy()
bdf['Column2'] *= 100  # Multiply to MVA
bdf[['Column3','Column4']] /= 100
bdf['Column4'] *= -1
bdf = bdf.rename(columns={'Column2': 'pf', 'Column3': 'mu_f', 'Column4': 'mu_t'})

pg.to_csv('pg.csv')
lmp.to_csv('lmp.csv')
bdf.to_csv('bdf.csv')
