from setup import *


class Cluster:

    def __init__(self, project_id, init_mode):

        self.project_id = project_id
        self.init_mode = init_mode

        oadir = path + '/data/spines/oa11'  # oa spine path
        adir = path + '/data/axis/' + self.project_id
        cdir = path + '/data/clustering/' + self.project_id
        nadir = path + '/data/initial_nodes/' + self.project_id
        dodir = path + '/data/discrimination_features/' + self.project_id  # discrimination features output path

        nasdir = path + '/data/node_assignments/' + self.project_id
        ncsdir = path + '/data/node_centres/' + self.project_id
        nfsdir = path + '/data/node_frequencies/' + self.project_id

        colour = ['blue','red','yellow','cyan','magenta','green','brown','gold','lawngreen','pink','gray','orange','indigo','salmon','chocolate',
                   'khaki','deepskyblue','lime','royalblue','wheat','deeppink','plum','olivedrab','teal','tomato','turquoise','rosybrown']

        pp = PdfPages(cdir + '/clustering.pdf')  # open model_outputs pdf for writing

        grid = pd.read_csv(oadir + '/oa11_spine.csv')

        files = os.listdir(adir)
        for f in files:
            a = pd.read_csv(adir + '/' + f)
            grid = pd.merge(a, grid, how='inner', left_on=['oa11'], right_on=['oa11'])  # join axis to oa spine

        # get oa11 index and delete oa11 from grid for use as array and oa11 stored against index
        oa11 = pd.DataFrame(grid['oa11'], columns=['oa11'])
        grid.drop(columns=['oa11'], inplace=True)
        grid_array = np.array(grid)
        grid_array_z = np.array(ss.zscore(grid_array))


        #k-means class arguments
        clusdf = grid_array_z
        n_init = 1
        max_iter = 100000
        n_clusters = 27
        init = [
         [-1,-1,-1]
        ,[-1,-1,0]
        ,[-1,-1,1]
        ,[-1,0,-1]
        ,[-1,0,0]
        ,[-1,0,1]
        ,[-1,1,-1]
        ,[-1,1,0]
        ,[-1,1,1]
        ,[0,-1,-1]
        ,[0,-1,0]
        ,[0,-1,1]
        ,[0,0,-1]
        ,[0,0,0]
        ,[0,0,1]
        ,[0,1,-1]
        ,[0,1,0]
        ,[0,1,1]
        ,[1,-1,-1]
        ,[1,-1,0]
        ,[1,-1,1]
        ,[1,0,-1]
        ,[1,0,0]
        ,[1,0,1]
        ,[1,1,-1]
        ,[1,1,0]
        ,[1,1,1]
        ]
        init = np.array(init)

        if self.init_mode == 'specified':
            initial_nodes = pd.read_csv(nadir + '/initial_nodes.csv')
            gazd = pd.DataFrame(grid_array_z, columns=(['A0', 'A1', 'A2']))
            init_nodes = pd.merge(oa11, gazd, left_index=True, right_index=True)
            init_nodes = pd.merge(initial_nodes, init_nodes, how='inner', left_on=['oa11'], right_on=['oa11'])
            init_nodes.drop(columns=['oa11'], inplace=True)
            init_nodes = np.array(init_nodes)
            print(init_nodes)
            init = init_nodes

        # assign each oa11 to nearest initial node (nearest neighbour lookup)
        nearest_node = np.zeros((len(grid_array_z), 4))
        nearest_node[:, :-1] = grid_array_z

        for row in range(len(nearest_node)):
            pt = nearest_node[row, [0, 1, 2]]
            dist, ind = spatial.KDTree(init).query(pt)
            nearest_node[row, 3] = ind

        # frequency of initial node allocations
        plt.hist(nearest_node[:, 3], bins=n_clusters)
        plt.title("""project_id = """ + self.project_id + """ : nearest node assignments : cluster nodes = """ + str(n_clusters),
                  fontsize=8)
        plt.xlabel('node')
        plt.ylabel('frequency')
        pp.savefig()

        # dataframe of initial node allocations
        nearest_node = pd.DataFrame(nearest_node[:, 3], columns=['nearestnode'])

        # k-means algoritm class instantiation
        kmeans = cl.KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter)

        # fit model and output nodes
        nodes = kmeans.fit_predict(clusdf)

        # entropy
        nodes_freq = (ss.stats.itemfreq(nodes))[:, 1]
        entropy = (nodes_freq * np.log(nodes_freq)).sum()

        # plot node frequencies
        fig = plt.figure()
        plt.hist(nodes, bins=n_clusters)
        plt.title("""project_id = """ + self.project_id + """ : cluster nodes = """ + str(n_clusters) + """ : iterations = """ + str(
            max_iter) + """ : entropy = """ + str(int(entropy)), fontsize=8)
        plt.xlabel('node')
        plt.ylabel('frequency')
        pp.savefig()

        # cluster centres
        cluscentres = kmeans.cluster_centers_

        mds = md.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=161,
                     dissimilarity="euclidean")

        mdsc = mds.fit_transform(cluscentres)

        labels = [i for i in range(len(mdsc))]
        fig = plt.figure()
        plt.scatter(mdsc[:, 0], mdsc[:, 1], c='b', cmap=plt.cm.Spectral)
        for i in range(len(mdsc)):
            xy = (mdsc[i][0], mdsc[i][1])
            plt.annotate(labels[i], xy)
        plt.title("""project_id = """ + self.project_id + """ : "2-D multidimensional scaling of 3-D axis : clusters = """ + str(
            n_clusters) + """ : iterations = """ + str(max_iter) + """ : entropy = """ + str(int(entropy)), fontsize=8)
        plt.xlabel('reduced dimension 1')
        plt.ylabel('reduced dimension 2')
        pp.savefig()

        # cluster centres dataframe
        cluscentres = pd.DataFrame(cluscentres)

        # get cluster centres next to initialisation nodes
        initialisations = pd.DataFrame(init)
        cluscentres = pd.merge(initialisations, cluscentres, left_index=True, right_index=True)

        # write initial nodes and final nodes to pdf
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.axis('tight')
        (ax.table(cellText=cluscentres.values.round(2), colLabels=cluscentres.columns, loc='center', fontsize=6,
                  colWidths=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05])).scale(0.75, 0.75)
        # fig.tight_layout()
        pp.savefig()

        # index nodes assignments
        cluslabels = kmeans.labels_
        cluslabels = pd.DataFrame(cluslabels, columns=['initialCluster'])

        # join oa11 to nodes assignment on index
        np.testing.assert_array_equal(oa11['oa11'].count(), cluslabels['initialCluster'].count(),
                                      nearest_node['nearestnode'].count())
        node_assignments = pd.merge(oa11, cluslabels, left_index=True, right_index=True)
        node_assignments = pd.merge(node_assignments, nearest_node, left_index=True, right_index=True)
        node_assignments['project_id'] = self.project_id
        print(node_assignments)

        # node frequencies
        node_frequencies = pd.DataFrame(node_assignments.groupby(['initialCluster'])['oa11'].count())

        from pandasql import sqldf
        pysqldf = lambda q: sqldf(q, globals())

        # output node assignments and axis
        cluscentres['cluster'] = cluscentres.index
        node_assignments_nodes = node_assignments.copy()
        node_assignments_nodes = pd.merge(node_assignments_nodes, cluscentres, how='inner', left_on=['initialCluster'], right_on=['cluster'])
        node_assignments_nodes['axis0'] = node_assignments_nodes['0_y']
        node_assignments_nodes['axis1'] = node_assignments_nodes['1_y']
        node_assignments_nodes['axis2'] = node_assignments_nodes['2_y']
        node_assignments_nodes = node_assignments_nodes[['oa11', 'initialCluster', 'axis0', 'axis1', 'axis2']]

        # 3-D scatter of initial and final nodes
        fig = figure()
        ax = Axes3D(fig)

        for i in range(len(cluscentres)):  # plot each point + it's index as text above
            ax.scatter(cluscentres['0_x'][i], cluscentres['1_x'][i], cluscentres['2_x'][i], color='b')
            ax.text(cluscentres['0_x'][i], cluscentres['1_x'][i], cluscentres['2_x'][i], '%s' % (str(i)), size=10, zorder=1,
                    color='k')

        ax.set_xlabel('Ax1')
        ax.set_ylabel('Ax2')
        ax.set_zlabel('Ax3')

        plt.title("""project_id = """ + self.project_id + """ : initial nodes""")

        pp.savefig()

        fig = figure()
        ax = Axes3D(fig)

        for i in range(len(cluscentres)):  # plot each point + it's index as text above
            ax.scatter(cluscentres['0_y'][i], cluscentres['1_y'][i], cluscentres['2_y'][i], color='b')
            ax.text(cluscentres['0_y'][i], cluscentres['1_y'][i], cluscentres['2_y'][i], '%s' % (str(i)), size=10, zorder=1,
                    color='k')

        ax.set_xlabel('Ax1')
        ax.set_ylabel('Ax2')
        ax.set_zlabel('Ax3')

        plt.title("""project_id = """ + self.project_id + """ : cluster nodes = """ + str(n_clusters) + """ : iterations = """ + str(
            max_iter) + """ : entropy = """ + str(int(entropy)), fontsize=8)

        pp.savefig()

        # add cluster labels to z space scatter and sample with replacement
        grid_array_z_cluster = pd.DataFrame(grid_array_z)
        grid_array_z_cluster = pd.merge(grid_array_z_cluster, cluslabels, left_index=True, right_index=True)
        grid_array_z_cluster = np.array(grid_array_z_cluster)
        grid_array_z_cluster = grid_array_z_cluster[np.random.randint(grid_array_z_cluster.shape[0], size=10000), :]

        # 3-D scatter of sample of oa11 allocated to parent cluster
        fig = figure()
        ax = Axes3D(fig)

        for i in range(len(grid_array_z_cluster)):  # plot each point + it's index as text above
            ax.scatter(grid_array_z_cluster[i][0], grid_array_z_cluster[i][1], grid_array_z_cluster[i][2],
                       color=colour[int(grid_array_z_cluster[i][3])])

        ax.set_xlabel('Ax1')
        ax.set_ylabel('Ax2')
        ax.set_zlabel('Ax3')

        plt.title("""project_id = """ + self.project_id + """ : points in z axis space by cluster""")

        pp.savefig()

        # 3-D scatter of sample of oa11 allocated to parent cluster
        fig = figure()
        ax = Axes3D(fig)

        for i in range(len(grid_array_z_cluster)):  # plot each point + it's index as text above
            ax.scatter(grid_array_z_cluster[i][0], grid_array_z_cluster[i][1], grid_array_z_cluster[i][2],
                       color=colour[int(grid_array_z_cluster[i][3])], alpha=0.2)

        ax.set_xlabel('Ax1')
        ax.set_ylabel('Ax2')
        ax.set_zlabel('Ax3')

        plt.title("""project_id = """ + self.project_id + """ : points in z axis space by cluster""")

        pp.savefig()



        # plot gains curves
        for file in os.listdir(dodir):
            print(file)
            input_feature = file[:-4]
            dc = pd.read_csv(dodir + '/' + input_feature + '.csv')
            ng = dc['order'].nunique()
            gini_grid_feature = pd.merge(dc, node_assignments, how='inner', left_on=['oa11'], right_on=['oa11'])
            tot = gini_grid_feature.groupby(['order']).agg({'metric': ['sum']}).reset_index()
            tot.columns = ["_".join(x) for x in tot.columns.ravel()]
            tot['order'] = tot['order_']
            tot['metric_tot'] = tot['metric_sum']
            tot = tot[['order', 'metric_tot']]
            gini_grid_feature = gini_grid_feature.groupby(['initialCluster', 'order']).agg({'metric': ['sum']}).reset_index()
            gini_grid_feature.columns = ["_".join(x) for x in gini_grid_feature.columns.ravel()]
            gini_grid_feature['initialCluster'] = gini_grid_feature['initialCluster_']
            gini_grid_feature['order'] = gini_grid_feature['order_']
            gini_grid_feature = pd.merge(gini_grid_feature, tot, how='inner', left_on=['order'],right_on=['order'])
            gini_grid_feature['metric_sum'] = gini_grid_feature['metric_sum'] / gini_grid_feature['metric_tot']
            gini_grid_feature = (gini_grid_feature.pivot(index='initialCluster', columns='order', values='metric_sum')).reset_index()

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

            plt.title("""project_id = """ + project_id + """ """ + input_feature + """ income gains curves""", fontsize=8)
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


            vols = (node_assignments.groupby(['initialCluster']).agg({'initialCluster': ['count']})).reset_index()
            vols.columns = ["_".join(x) for x in vols.columns.ravel()]
            vols['volume'] = vols['initialCluster_count']
            vols['initialCluster'] = vols['initialCluster_']
            print(vols)

            for j in range(1, ng + 1):
                gini_grid_working = gini_grid_feature.iloc[:, [0, j]]
                gini_grid_working['grouping'] = gini_grid_working.iloc[:, [1]]

                gini_base = pd.merge(gini_grid_working, vols, how='inner', left_on=['initialCluster'],right_on=['initialCluster'])
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
            plt.title("""project_id = """ + project_id + """gini by """ + input_feature + """ income group""", fontsize=8)
            plt.xlabel('gini')
            plt.ylabel(input_feature + ' group')
            pp.savefig()
            plt.show()

        pp.close()

        # output node assignments,cluster centres and frequencies to file
        node_assignments.to_csv(nasdir + """/node_assignments.csv""", index=False)
        cluscentres.to_csv(ncsdir + """/node_centres.csv""", index=False)
        node_frequencies.to_csv(nfsdir + """/node_frequencies.csv""", index=True)
