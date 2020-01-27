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
        dodir = path + '/data/discrimination_features/' + self.project_id  # discrimination features output path

        md = pd.read_csv(mdir + '/feature_categorisation.csv')  # read in feature categorisation metadata file
        md = md[md['discrimination_feature'] != 1]
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










        # plot gains curves
        for axfile in os.listdir(adir):
            for file in os.listdir(dodir):
                print(axfile)
                print(file)
                ax = axfile[:-4]

                axis = pd.read_csv(adir + '/' + ax + '.csv')
                axis['axis'] = axis[ax]
                axis['initialCluster'] = pd.qcut(axis['axis'], 10, labels=False, duplicates='drop')

                input_feature = file[:-4]
                dc = pd.read_csv(dodir + '/' + input_feature + '.csv')
                ng = dc['order'].nunique()

                gini_grid_feature = pd.merge(axis, dc, how='inner', left_on=['oa11'], right_on=['oa11'])
                tot = gini_grid_feature.groupby(['order']).agg({'metric': ['sum']}).reset_index()
                tot.columns = ["_".join(x) for x in tot.columns.ravel()]
                tot['order'] = tot['order_']
                tot['metric_tot'] = tot['metric_sum']
                tot = tot[['order', 'metric_tot']]
                gini_grid_feature = gini_grid_feature.groupby(['initialCluster', 'order']).agg(
                    {'metric': ['sum']}).reset_index()
                gini_grid_feature.columns = ["_".join(x) for x in gini_grid_feature.columns.ravel()]
                gini_grid_feature['initialCluster'] = gini_grid_feature['initialCluster_']
                gini_grid_feature['order'] = gini_grid_feature['order_']
                gini_grid_feature = pd.merge(gini_grid_feature, tot, how='inner', left_on=['order'], right_on=['order'])
                gini_grid_feature['metric_sum'] = gini_grid_feature['metric_sum'] / gini_grid_feature['metric_tot']
                gini_grid_feature = (
                    gini_grid_feature.pivot(index='initialCluster', columns='order', values='metric_sum')).reset_index()

                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)

                for i in range(1, ng + 1):
                    gini_grid_working = gini_grid_feature.iloc[:, [0, i]]
                    gini_grid_working['grouping'] = gini_grid_working.iloc[:, [1]]
                    gini_grid_working = gini_grid_working.sort_values(by=['grouping'], ascending=False)
                    gini_grid_feature_decile = gini_grid_working['initialCluster']
                    del gini_grid_working['initialCluster']
                    gini_grid_feature_cumsum = gini_grid_working.cumsum()
                    gini_grid_feature_decile = pd.DataFrame(gini_grid_feature_decile)
                    gini_grid_feature_cumsum = pd.DataFrame(gini_grid_feature_cumsum)
                    gini_grid_feature_gains = pd.merge(gini_grid_feature_decile, gini_grid_feature_cumsum, left_index=True,
                                                       right_index=True)
                    gini_grid_feature_gains = gini_grid_feature_gains.reset_index()
                    gini_grid_feature_gains['order'] = gini_grid_feature_gains.index

                    ax1.plot(gini_grid_feature_gains.iloc[:, [4]], gini_grid_feature_gains.iloc[:, [2]])

                    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
                colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
                for i, j in enumerate(ax1.lines):
                    j.set_color(colors[i])

                plt.title("""project_id = """ + project_id + """ """ + input_feature + """ income gains curves""",
                          fontsize=8)
                plt.xlabel('cluster rank')
                plt.ylabel('cumulatiuve frequency')

                ax1.legend(loc=0)
                pp.savefig()
                plt.show()

                # income gini
                # group headers
                gini_list = []
                grouping = list(gini_grid_feature)
                del grouping[0]
                # volumne by cluster

                vols = (axis.groupby(['initialCluster']).agg({'initialCluster': ['count']})).reset_index()
                vols.columns = ["_".join(x) for x in vols.columns.ravel()]
                vols['volume'] = vols['initialCluster_count']
                vols['initialCluster'] = vols['initialCluster_']
                print(vols)

                for j in range(1, ng + 1):
                    gini_grid_working = gini_grid_feature.iloc[:, [0, j]]
                    gini_grid_working['grouping'] = gini_grid_working.iloc[:, [1]]

                    gini_base = pd.merge(gini_grid_working, vols, how='inner', left_on=['initialCluster'],
                                         right_on=['initialCluster'])
                    gini_base = gini_base[['initialCluster', 'volume', 'grouping']]
                    gini_base = gini_base.sort_values(['grouping'], ascending=[False])

                    tot_vol = gini_base['volume'].sum()
                    gini_base['volume'] = gini_base['volume'] / tot_vol
                    gini_base['cuml'] = gini_base['grouping'].cumsum()
                    total = 0
                    for i in range(len(gini_base)):
                        total += gini_base['volume'][i] * (gini_base['grouping'][i] + (2 * (1 - gini_base['cuml'][i])))
                    gini = 1 - total
                    print(gini)
                    gini_list += [gini]
                gini_list = pd.DataFrame(gini_list, columns=[input_feature + '_grouping'])
                grouping = pd.DataFrame(grouping, columns=['gini'])
                gini_output = pd.merge(grouping, gini_list, left_index=True, right_index=True)
                plt.barh(gini_output['gini'], gini_output[input_feature + '_grouping'], color="blue")
                plt.title("""project_id = """ + project_id + """gini by """ + input_feature + """ income group""",
                          fontsize=8)
                plt.xlabel('gini')
                plt.ylabel(input_feature + ' group')
                pp.savefig()
                plt.show()

        #close model outputs pdf
        pp.close()
