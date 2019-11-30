from setup import *

mdir = path + '/data/metadata/feature_categorisation.csv'  # feature categorisation metadata file
idir = path + '/data/input_tests'  # features input file path
odir = path + '/data/features'  # features output path

md = pd.read_csv(mdir)  # read in feature categorisation metadata file
fl = md['feature'].unique()

files = os.listdir(odir)
for f in files:
    os.remove(odir + '/' + f)

def features(feature, input_feature):

    print(input_feature)
    df = pd.read_csv(idir + '/' + str(input_feature) + '_cleaned.csv')  # read in feature input file
    df = df.head(1000)  # for dev
    mdi = md[md['feature'] == feature]
    df = pd.merge(df, mdi, how='inner', left_on=['level'], right_on=['level'])  # join feature input file to metadata

    # order oas by level and create cumulative sums of metric against each level
    df = df.sort_values(['oa11', 'order'], ascending=[True, True])
    df['cuml'] = df.groupby(['oa11'])['metric'].apply(lambda x: x.cumsum())

    # determine maximum cumulative value, number of levels and splitting level
    mx = (df.groupby(['oa11']).agg(max=('cuml', 'max'))).reset_index()
    ng = df['level'].nunique()
    ln = int(mdi[mdi['split'] == 1]['order'])

    # calculate cumulative proportions of maximum for each level
    df = pd.merge(df, mx, how='inner', left_on=['oa11'], right_on=['oa11'])
    df['cuml_prop'] = (df['cuml'] + (df['order'] / ng)) / (df['max'] + 1)

    # calculate log odds for the splitting level and write as csv to feature output directory
    df = df[df['order'] == ln]
    df[feature] = np.log(df['cuml_prop']) - (np.log(1 - (df['cuml_prop'])))
    df = df[['oa11', feature]]
    np.testing.assert_array_equal(df['oa11'].nunique(), df['oa11'].count())
    df.to_csv(odir + '/' + feature + '.csv', index=False)

for f in fl:
    mdi = md[md['feature'] == f]
    input_feature = mdi['input_feature'].unique()[0]
    feature = mdi['feature'].unique()[0]
    features(feature, input_feature)





