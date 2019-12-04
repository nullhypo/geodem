from setup import *


class Axis:

    # class attributes
    oadir = path + '/data/spines/oa11'  # oa spine path

    def __init__(self, project_id):

        self.project_id = project_id

        odir = path + '/data/features/' + self.project_id  # features output path
        ahdir = path + '/data/axis_histograms/' + self.project_id
        adir = path + '/data/axis/' + self.project_id
        mdir = path + '/data/metadata/' + self.project_id  # feature categorisation metadata file
        fdir = path + '/data/features_scatter/' + self.project_id

        md = pd.read_csv(mdir + '/feature_categorisation.csv')  # read in feature categorisation metadata file
        md = md[['feature', 'axis']]
        md.drop_duplicates(inplace=True)

        grid = pd.read_csv(self.oadir + '/oa11_spine.csv')

        features_set = list(md['feature'])
        for f in features_set:
            df = pd.read_csv(odir + '/' + f + '.csv')  # read in feature
            grid = pd.merge(df, grid, how='inner', left_on=['oa11'], right_on=['oa11'])  # progressively build grid

        # get oa11 index and delete oa11 from grid for use as array and oa11 stored against index
        oa11 = pd.DataFrame(grid['oa11'], columns=['oa11'])
        grid.drop(columns=['oa11'], inplace=True)

        gridc = np.array(grid)
        grid_array_z_c = ss.zscore(gridc)
        grid_array_z_c = grid_array_z_c[np.random.randint(grid_array_z_c.shape[0], size=1000), :]

        pp = PdfPages(fdir + '/features_scatter.pdf')  # open model_outputs pdf for writing

        fln = len(features_set)
        for i in range(0, fln):
            for j in range(0, fln):
                if i != j:
                    # features scatter
                    fig = plt.figure()
                    plt.scatter(grid_array_z_c[:, i], grid_array_z_c[:, j], c='b', cmap=plt.cm.Spectral, s=0.01)
                    plt.xlabel(features_set[i])
                    plt.ylabel(features_set[j])
                    plt.title('Features scatter')
                    pp.savefig()

        pp.close()

        pp = PdfPages(ahdir + '/axis_histograms.pdf')  # open model_outputs pdf for writing

        # PCA
        def PCA_analysis(grid, axis, feature_count):
            # instantiate PCA class on Axis features (normalised)
            print(grid.shape)
            pca = PCA(n_components=1)
            # get primary eigenvector of covariance matrix
            pca.fit(grid)
            v1 = pca.components_
            # project features space onto the primary eigenvector
            PC['pcDataPoint'] = pca.transform(grid)

        # histogram
        def hist(axis, source):
            fig = plt.figure()
            plt.hist(source, bins=100)
            plt.title("""Axis """ + str(axis) + """ histogram""", fontsize=16)
            plt.xlabel("""Axis """ + str(axis))
            plt.ylabel('frequency')
            pp.savefig()

        # output axis def
        def PCA_out(axis, source, feature_count):
            pcDataPoint = pd.DataFrame(source, columns=["""Ax""" + str(axis)])
            pcDataPoint = pd.merge(oa11, pcDataPoint, left_index=True, right_index=True)
            pcDataPoint.to_csv(adir + '/Ax' + str(axis) + '.csv',index=False)

        # create axis using PCA
        for i in range(1, 4):
            print("""i = """ + str(i))
            features = []
            for j in features_set:
                print("""j = """ + str(j))
                mdi = md[md['feature'] == j]
                if int(mdi['axis']) == i:
                    features += [j]
            feature_count = len(features)
            print("""feature count =""" + str(feature_count))
            print("""features = """ + str(features))

            grid_array = grid[features]
            grid_array = np.array(grid_array)

            # standardise log odds features (or feature is oa11 unique)
            grid_array_z_features = ss.zscore(grid_array)
            print(grid_array_z_features.shape)

            if feature_count == 1:
                source = grid_array_z_features
                print("""source""")
                print(source)
                print(source.shape)
            else:
                PC = {}
                PCA_analysis(grid_array_z_features, i, feature_count)
                source = PC['pcDataPoint']
                print("""source""")
                print(source)
                print(source.shape)

            # merge principle eigenvector back to oa11 level and export
            PCA_out(i, source, feature_count)
            # histogram axis
            hist(i, source)

        #close model outputs pdf
        pp.close()
