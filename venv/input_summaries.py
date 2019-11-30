from setup import *
lm = pd.DataFrame()
sm = pd.DataFrame()
dir = path + '/data/inputs'
odir = path + '/data/input_summaries'
pp = PdfPages(odir + '/level_summaries.pdf')

for file in os.listdir(dir):
    print(file)
    feature = file[:-4]

    df = pd.read_csv(dir + '/' + str(file))
    rc = df['oa11'].count()
    oc = df['oa11'].nunique()
    lc = df['level'].nunique()
    ss = pd.DataFrame({'feature': feature, 'row_count': rc, 'oa11_count': oc, 'levels_count': lc}, index=[0])
    sm = sm.append(ss)

    ls = (df.groupby(['level']).agg(count=('metric', 'count'),sum=('metric', 'sum'),min=('metric', 'min'),max=('metric', 'max'))).reset_index()
    ls['feature'] = feature
    lm = lm.append(ls)

    fig = plt.figure(figsize=(10, 4))
    plt.barh(ls['level'], ls['sum'], align='center', alpha=0.5)
    plt.yticks(ls['level'])
    plt.xlabel('metric')
    plt.title(feature)
    plt.tight_layout()
    pp.savefig()

pp.close()

sm.to_csv(odir + '/top_line_summary.csv', index=False)
lm.to_csv(odir + '/level_summaries.csv', index=False)
