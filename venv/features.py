from setup import *


class Features:

    # class attributes
    oadir = path + '/data/spines/oa11'

    def __init__(self, project_id):

        self.project_id = project_id

        mdir = path + '/data/metadata/' + self.project_id  # feature categorisation metadata file
        idir = path + '/data/inputs/' + self.project_id  # features input file path
        odir = path + '/data/features/' + self.project_id  # features output path
        dodir = path + '/data/discrimination_features/' + self.project_id  # discrimination features output path
        lohdir = path + '/data/log_odds_histograms/' + self.project_id

        pp = PdfPages(lohdir + '/log_odds_histograms.pdf')  # open model_outputs pdf for writing
        md = pd.read_csv(mdir + '/feature_categorisation.csv')  # read in feature categorisation metadata file
        fl = md['feature'].unique()

        oa = pd.read_csv(self.oadir + '/oa11_spine.csv')

        files = os.listdir(odir)
        for f in files:
            os.remove(odir + '/' + f)

        files = os.listdir(dodir)
        for f in files:
            os.remove(dodir + '/' + f)

        def features(feature, input_feature):

            print(input_feature)
            df = pd.read_csv(idir + '/' + str(input_feature) + '_cleaned.csv')  # read in feature input file
            df = pd.merge(df, oa, how='inner', left_on=['oa11'], right_on=['oa11'])  # join feature input file to oa spine
            #df = df.head(1000)  # for dev
            mdi = md[md['feature'] == feature]
            df = pd.merge(df, mdi, how='inner', left_on=['level'], right_on=['level'])  # join feature input file to metadata

            discrim_flag = df['discrimination_feature'].unique()[0]
            if discrim_flag != 1:

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

                # histogram of log odds
                fig = plt.figure()
                plt.hist(df[feature], bins=100)
                plt.title(feature + ' histogram', fontsize=16)
                plt.xlabel(feature)
                plt.ylabel('frequency')
                pp.savefig()

            else:
                dfd = df.copy()
                dfd = dfd[['oa11', 'order', 'metric']]
                dfd = dfd.sort_values(['oa11', 'order'], ascending=[True, True])
                dfd.to_csv(dodir + '/' + input_feature + '.csv', index=False)

        # Loop through features in metadata file and run features function for each
        for f in fl:
            mdi = md[md['feature'] == f]
            input_feature = mdi['input_feature'].unique()[0]
            feature = mdi['feature'].unique()[0]
            features(feature, input_feature)

        pp.close()





