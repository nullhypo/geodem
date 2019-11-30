from setup import *



mdir = path + '/data/metadata/feature_categorisation.csv'
idir = path + '/data/inputs'
odir = path + '/data/features'

md = pd.read_csv(mdir)


file = 'oa11_Age_Adult_cleaned.csv'
#for file in os.listdir(dir):
print(file)
feature = file[:-12]

df = pd.read_csv(idir + '/' + str(file))

df = df.head(1000)

df = pd.merge(df, md, how='inner', left_on=['level'], right_on=['level'])

df = df.sort_values(['oa11', 'order'], ascending=[True, True])
df['cuml'] = df.groupby(['oa11'])['metric'].apply(lambda x: x.cumsum())
mx = (df.groupby(['oa11']).agg(max=('cuml', 'max'))).reset_index()
ng = df['level'].nunique()
ln = int(md[md['split'] == 1]['order'])
df = pd.merge(df, mx, how='inner', left_on=['oa11'], right_on=['oa11'])
df['cuml_prop'] = (df['cuml'] + (df['order'] / ng)) / (df['max'] + 1)
df = df[df['order'] == ln]
df[feature] = np.log(df['cuml_prop']) - (np.log(1 - (df['cuml_prop'])))
df = df[['oa11', feature]]

df.to_csv(odir + '/' + feature + '.csv', index=False)




