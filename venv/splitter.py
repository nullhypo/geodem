from setup import *


class Splitter:

    def __init__(self, project_id):

        self.project_id = project_id

        sdir = path + '/data/splitting'

        oadir = path + '/data/spines/oa11'  # oa spine path
        adir = path + '/data/axis/' + self.project_id
        cdir = path + '/data/clustering/' + self.project_id
        nadir = path + '/data/initial_nodes/' + self.project_id
        dodir = path + '/data/discrimination_features/' + self.project_id  # discrimination features output path

        nasdir = path + '/data/node_assignments/' + self.project_id
        ncsdir = path + '/data/node_centres/' + self.project_id
        nfsdir = path + '/data/node_frequencies/' + self.project_id
        subdir = path + '/data/metadata/' + self.project_id

        colour = ['blue', 'red', 'yellow', 'cyan', 'magenta', 'green', 'brown', 'gold', 'lawngreen', 'pink', 'gray',
                  'orange', 'indigo', 'salmon', 'chocolate',
                  'khaki', 'deepskyblue', 'lime', 'royalblue', 'wheat', 'deeppink', 'plum', 'olivedrab', 'teal',
                  'tomato', 'turquoise', 'rosybrown','blue', 'red', 'yellow', 'cyan', 'magenta', 'green', 'brown', 'gold', 'lawngreen', 'pink', 'gray',
                  'orange', 'indigo', 'salmon', 'chocolate',
                  'khaki', 'deepskyblue', 'lime', 'royalblue', 'wheat', 'deeppink', 'plum', 'olivedrab', 'teal',
                  'tomato', 'turquoise', 'rosybrown','blue', 'red', 'yellow', 'cyan', 'magenta', 'green', 'brown', 'gold', 'lawngreen', 'pink', 'gray',
                  'orange', 'indigo', 'salmon', 'chocolate',
                  'khaki', 'deepskyblue', 'lime', 'royalblue', 'wheat', 'deeppink', 'plum', 'olivedrab', 'teal',
                  'tomato', 'turquoise', 'rosybrown']

        pp = PdfPages(sdir + '/splitting.pdf')  # open model_outputs pdf for writing

        # get node assignments, parent node centres and subcluster metadata
        node_assignments = pd.read_csv(nasdir + """/node_assignments.csv""")
        node_centres = pd.read_csv(ncsdir + """/node_centres.csv""")
        subcluster_metadata = pd.read_csv(subdir + """/subcluster_metadata.csv""")

        nodes_merged = psql.sqldf("""
            select   na.oa11
                    ,na.initialCluster
                    ,cast(coalesce(sm.merge,na.initialCluster) as int) as mergedCluster
            from node_assignments na
            left outer join subcluster_metadata sm
            on na.initialCluster = sm.initialCluster
            ;
            """, locals())

        # join axis into a master grid and append mergeCluster
        Ax = {}

        Axis = {
            0: ["dummy", "dummy", "Ax1"]
            , 1: ["dummy", "dummy", "Ax2"]
            , 2: ["dummy", "dummy", "Ax3"]
        }

        for axis in Axis:

            Ax[axis] = pd.read_csv(adir + """/""" + Axis[axis][2] + """.csv""")

        df0 = Ax[0]
        df1 = Ax[1]
        df2 = Ax[2]

        df0['Ax0'] = df0['Ax1']
        df0.drop(columns=['Ax1'])
        df1['Ax1'] = df1['Ax2']
        df1.drop(columns=['Ax2'])
        df2['Ax2'] = df2['Ax3']
        df2.drop(columns=['Ax3'])

        grid_master = psql.sqldf("""
            select   df0.oa11
                    ,df0.Ax0
                    ,df1.Ax1
                    ,df2.Ax2 
                    ,nm.initialCluster
                    ,nm.mergedCluster
            from df0 df0
            inner join df1 df1
            on df0.oa11 = df1.oa11
            inner join df2 df2
            on df0.oa11 = df2.oa11
            inner join nodes_merged nm
            on df0.oa11 = nm.oa11
            ;
            """, locals())

        subcluster_merged_metadata = subcluster_metadata[pd.isnull(subcluster_metadata['merge'])]
        subcluster_merged_metadata = subcluster_merged_metadata.reset_index()
        node_assignments_split = pd.DataFrame()
        cluscentres_split = pd.DataFrame()

        for i in range(len(subcluster_merged_metadata)):

            parent_node = subcluster_merged_metadata['initialCluster'][i]
            n_clusters = subcluster_merged_metadata['split'][i]

            if subcluster_merged_metadata['split'][i] > 1:

                # grid for parent node
                grid = psql.sqldf("""
                    select   oa11
                            ,ax0
                            ,ax1
                            ,ax2
                    from grid_master
                    where mergedCluster = """ + str(parent_node) + """
                    ;
                    """, locals())

                # get oa11 index and delete oa11 from grid  for use as array and oa11 stored against index
                oa11 = pd.DataFrame(grid['oa11'], columns=['oa11'])
                grid.drop(columns=['oa11'], inplace=True)

                # convert grid to array
                grid_array_z = np.array(grid)
                # grid_array_z = np.array(ss.zscore(grid_array))

                # axis histograms of log odds for parent subset
                for axis in Axis:
                    plt.hist(grid_array_z[:, axis], bins=100)
                    plt.title(Axis[axis][2] + """ : """ + Axis[axis][1], fontsize=16)
                    plt.xlabel('Z log odds')
                    plt.ylabel('frequency')
                    pp.savefig()
                    plt.show()

                meanaxis = []
                for axis in Axis:
                    meanaxis += [grid_array_z[:, axis].mean()]
                meanaxis = pd.DataFrame(meanaxis)
                meanaxis

                # meanaxis = node_centres[['0_y','1_y','2_y']].iloc[parent_node]
                # meanaxis=pd.DataFrame(meanaxis)

                # write parent nodes centre to pdf
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.axis('tight')
                (ax.table(cellText=meanaxis.values.round(2), colLabels=meanaxis.columns, loc='center', fontsize=6,
                          colWidths=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05])).scale(0.75, 0.75)
                # fig.tight_layout()
                pp.savefig()
                plt.show()

                if n_clusters == 2:
                    # 2 cluster initalisation centres
                    p_35 = []
                    p_65 = []
                    for axis in Axis:
                        p_35 += [np.percentile(grid_array_z[:, axis], 35)]
                        p_65 += [np.percentile(grid_array_z[:, axis], 65)]
                    init = np.array([np.array(p_35), np.array(p_65)])
                    # 3 cluster initalisation centres
                elif n_clusters == 3:
                    p_25 = []
                    p_50 = []
                    p_75 = []
                    for axis in Axis:
                        p_25 += [np.percentile(grid_array_z[:, axis], 25)]
                        p_50 += [np.percentile(grid_array_z[:, axis], 50)]
                        p_75 += [np.percentile(grid_array_z[:, axis], 75)]
                    init = np.array([np.array(p_25), np.array(p_50), np.array(p_75)])

                    # k-means class arguments
                clusdf = grid_array_z
                n_init = 1
                max_iter = 100

                # k-means algoritm class instantiation
                kmeans = cl.KMeans(n_clusters=n_clusters, n_init=n_init, init=init, max_iter=max_iter)

                # fit model and output nodes
                nodes = kmeans.fit_predict(clusdf)

                # entropy
                nodes_freq = (ss.stats.itemfreq(nodes))[:, 1]
                entropy = (nodes_freq * np.log(nodes_freq)).sum()
                entropy

                # plot node frequencies
                plt.hist(nodes, bins=int(n_clusters))
                plt.title("""project_id = """ + project_id + """ : parent node = """ + str(
                    parent_node) + """: cluster nodes = """ + str(n_clusters) + """ : n init = """ + str(
                    n_init) + """ iterations = """ + str(max_iter) + """ : entropy = """ + str(int(entropy)),
                          fontsize=8)
                plt.xlabel('node')
                plt.ylabel('frequency')
                pp.savefig()
                plt.show()

                # cluster centres
                cluscentres = kmeans.cluster_centers_
                cluscentres = pd.DataFrame(cluscentres)

                # index nodes assignments
                cluslabels = kmeans.labels_
                cluslabels = pd.DataFrame(cluslabels, columns=['childCluster'])
                cluslabels['mergedCluster'] = parent_node
                cluslabels

                grid_array_z_childsplit = pd.merge(grid, cluslabels, left_index=True, right_index=True)

                scat_colour = ['b', 'g', 'r']

                # 3-D scatter of oa11 allocated to parent cluster
                fig = figure()
                ax = Axes3D(fig)

                for i in range(len(grid_array_z_childsplit)):  # plot each point + it's index as text above
                    ax.scatter(grid_array_z_childsplit['Ax0'][i], grid_array_z_childsplit['Ax1'][i],
                               grid_array_z_childsplit['Ax2'][i],
                               color=scat_colour[grid_array_z_childsplit['childCluster'][i]])

                ax.set_xlabel(Axis[0][2] + """ : """ + Axis[0][1])
                ax.set_ylabel(Axis[1][2] + """ : """ + Axis[1][1])
                ax.set_zlabel(Axis[2][2] + """ : """ + Axis[2][1])

                plt.title("""project_id = """ + project_id + """ : parent node """ + str(
                    parent_node) + """ allocated points scatter""")

                pp.savefig()
                plt.show()

                # get parent node centres as dataframre
                initialisations = pd.DataFrame(init)  # [['0_y','1_y','2_y']].iloc[parent_node]).T

                cluscentres_init = pd.merge(cluscentres, initialisations, left_index=True, right_index=True)

                # write initial nodes and final nodes to pdf
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.axis('tight')
                (ax.table(cellText=cluscentres_init.values.round(2), colLabels=cluscentres_init.columns, loc='center',
                          fontsize=6, colWidths=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05])).scale(0.75, 0.75)
                # fig.tight_layout()
                pp.savefig()

                # join oa11 to nodes assignent on index
                np.testing.assert_array_equal(oa11['oa11'].count(), cluslabels['mergedCluster'].count())
                node_assignments_sc = pd.merge(oa11, cluslabels, left_index=True, right_index=True)
                node_assignments_sc['project_id'] = project_id
                node_assignments_sc

                # 3-D scatter of initial and final nodes
                fig = figure()
                ax = Axes3D(fig)

                for i in range(len(cluscentres_init)):  # plot each point + it's index as text above
                    ax.scatter(cluscentres_init['0_y'][i], cluscentres_init['1_y'][i], cluscentres_init['2_y'][i],
                               color='b')
                    ax.text(cluscentres_init['0_y'][i], cluscentres_init['1_y'][i], cluscentres_init['2_y'][i],
                            '%s' % (str(i)), size=10, zorder=1, color='k')

                ax.set_xlabel(Axis[0][2] + """ : """ + Axis[0][1])
                ax.set_ylabel(Axis[1][2] + """ : """ + Axis[1][1])
                ax.set_zlabel(Axis[2][2] + """ : """ + Axis[2][1])

                plt.title(
                    """project_id = """ + project_id + """ : parent node """ + str(parent_node) + """ initial nodes""")

                pp.savefig()
                plt.show()

                fig = figure()
                ax = Axes3D(fig)

                for i in range(len(cluscentres_init)):  # plot each point + it's index as text above
                    ax.scatter(cluscentres_init['0_x'][i], cluscentres_init['1_x'][i], cluscentres_init['2_x'][i],
                               color='b')
                    ax.text(cluscentres_init['0_x'][i], cluscentres_init['1_x'][i], cluscentres_init['2_x'][i],
                            '%s' % (str(i)), size=10, zorder=1, color='k')

                ax.set_xlabel(Axis[0][2] + """ : """ + Axis[0][1])
                ax.set_ylabel(Axis[1][2] + """ : """ + Axis[1][1])
                ax.set_zlabel(Axis[2][2] + """ : """ + Axis[2][1])

                plt.title("""project_id = """ + project_id + """ : parent node """ + str(
                    parent_node) + """child cluster nodes = """ + str(n_clusters) + """ : iterations = """ + str(
                    max_iter) + """ : entropy = """ + str(int(entropy)), fontsize=8)

                pp.savefig()
                plt.show()

                cluscentres['initialCluster'] = parent_node
                cluscentres_split = cluscentres_split.append(cluscentres)

            else:

                node_assignments_sc = psql.sqldf("""
                        select   oa11
                                ,0 as childCluster
                                ,mergedCluster
                                ,'""" + project_id + """' as project_id
                        from nodes_merged
                        where mergedCluster = """ + str(parent_node) + """
                        ;
                        """, locals())

            node_assignments_split = node_assignments_split.append(node_assignments_sc)

        # create segment code from parent and child nodes
        node_assignments_split = psql.sqldf("""
                        select   *
                                ,cast(mergedCluster as varchar)||'_'||cast(childCluster as varchar) as segment
                        from node_assignments_split
                        ;
                        """, locals())

        cluscentres_split['childCluster'] = cluscentres_split.index

        cluscentres_split = psql.sqldf("""
                        select   ndc.cluster as initialCluster
                                ,ndc.[0_y] as initialCluster_ax0
                                ,ndc.[1_y] as initialCluster_ax1
                                ,ndc.[2_y] as initialCluster_ax2
                                ,cls.childCluster
                                ,case when cls.[0] is null then ndc.[0_y] else cls.[0] end as finalCluster_ax0
                                ,case when cls.[1] is null then ndc.[1_y] else cls.[1] end as finalCluster_ax1
                                ,case when cls.[2] is null then ndc.[2_y] else cls.[2] end as finalCluster_ax2
                                ,case when cls.childCluster is null then cast(cluster as varchar) else cast(initialCluster as varchar)||'_'||cast(childCluster as varchar) end as segment
                        from node_centres ndc
                        left outer join cluscentres_split cls
                        on ndc.cluster = cls.initialCluster
                        ;
                        """, locals())

        # plot node frequencies
        plt.hist(node_assignments_split['segment'], bins=len(cluscentres_split))
        plt.title("""project_id = """ + project_id + """ : cluster nodes = """ + str(len(cluscentres_split)))
        plt.xlabel('node')
        plt.ylabel('frequency')
        pp.savefig()
        plt.show()

        mds = md.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=161,
                     dissimilarity="euclidean")

        mdsc = mds.fit_transform(node_centres[['0_y', '1_y', '2_y']])

        # labels = [i for i in range(len(mdsc))]
        labels = node_centres['cluster']
        plt.scatter(mdsc[:, 0], mdsc[:, 1], c='b', cmap=plt.cm.Spectral)
        for i in range(len(mdsc)):
            xy = (mdsc[i][0], mdsc[i][1])
            plt.annotate(labels[i], xy, fontsize=10)
        plt.title("""project_id = """ + project_id + """ : "2-D multidimensional scaling of 3-D axis : clusters = """ + str(
            len(cluscentres_split)))
        plt.xlabel('reduced dimension 1')
        plt.ylabel('reduced dimension 2')
        plt.plot()
        pp.savefig()
        plt.show()

        mdsc = mds.fit_transform(cluscentres_split[['finalCluster_ax0', 'finalCluster_ax1', 'finalCluster_ax2']])

        # labels = [i for i in range(len(mdsc))]
        labels = cluscentres_split['segment']
        plt.scatter(mdsc[:, 0], mdsc[:, 1], c='b', cmap=plt.cm.Spectral)
        for i in range(len(mdsc)):
            xy = (mdsc[i][0], mdsc[i][1])
            plt.annotate(labels[i], xy)
        plt.title("""project_id = """ + project_id + """ : "2-D multidimensional scaling of 3-D axis : clusters = """ + str(
            len(cluscentres_split)))
        plt.xlabel('reduced dimension 1')
        plt.ylabel('reduced dimension 2')
        plt.plot()
        pp.savefig()
        plt.show()

        # 3-D scatter of initial and final nodes
        fig = figure()
        ax = Axes3D(fig)

        for i in range(len(node_centres)):  # plot each point + it's index as text above
            ax.scatter(node_centres['0_y'][i], node_centres['1_y'][i], node_centres['2_y'][i], color='b')
            ax.text(node_centres['0_y'][i], node_centres['1_y'][i], node_centres['2_y'][i], '%s' % (str(i)), size=10,
                    zorder=1, color='k')

        ax.set_xlabel(Axis[0][2] + """ : """ + Axis[0][1])
        ax.set_ylabel(Axis[1][2] + """ : """ + Axis[1][1])
        ax.set_zlabel(Axis[2][2] + """ : """ + Axis[2][1])

        plt.title("""project_id = """ + project_id + """ : initial nodes""")

        pp.savefig()
        plt.show()

        fig = figure()
        ax = Axes3D(fig)

        for i in range(len(cluscentres_split)):  # plot each point + it's index as text above
            ax.scatter(cluscentres_split['finalCluster_ax0'][i], cluscentres_split['finalCluster_ax1'][i],
                       cluscentres_split['finalCluster_ax2'][i], color='b')
            ax.text(cluscentres_split['finalCluster_ax0'][i], cluscentres_split['finalCluster_ax1'][i],
                    cluscentres_split['finalCluster_ax2'][i], '%s' % (str(i)), size=10, zorder=1, color='k')

        ax.set_xlabel(Axis[0][2] + """ : """ + Axis[0][1])
        ax.set_ylabel(Axis[1][2] + """ : """ + Axis[1][1])
        ax.set_zlabel(Axis[2][2] + """ : """ + Axis[2][1])

        plt.title("""project_id = """ + project_id + """ : final nodes = """ + str(len(cluscentres_split)))

        pp.savefig()
        plt.show()

        fig = figure()
        ax = Axes3D(fig)

        for i in range(len(cluscentres_split)):  # plot each point + it's index as text above
            ax.scatter(cluscentres_split['finalCluster_ax0'][i], cluscentres_split['finalCluster_ax1'][i],
                       cluscentres_split['finalCluster_ax2'][i], color='b')
            ax.text(cluscentres_split['finalCluster_ax0'][i], cluscentres_split['finalCluster_ax1'][i],
                    cluscentres_split['finalCluster_ax2'][i], '%s' % (cluscentres_split['segment'][i]), size=10, zorder=1,
                    color='k')

        ax.set_xlabel(Axis[0][2] + """ : """ + Axis[0][1])
        ax.set_ylabel(Axis[1][2] + """ : """ + Axis[1][1])
        ax.set_zlabel(Axis[2][2] + """ : """ + Axis[2][1])

        plt.title("""project_id = """ + project_id + """ : final nodes = """ + str(len(cluscentres_split)))

        pp.savefig()
        plt.show()

        # append arbitary segment label and axis
        dist_segs = psql.sqldf("""
                        select   distinct segment
                        from node_assignments_split
                        ;
                        """, locals())
        dist_segs['seg'] = dist_segs.index
        grid_array_z_cluster = psql.sqldf("""
                        select   gms.Ax0
                                ,gms.Ax1
                                ,gms.Ax2
                                ,cast(dsg.seg as int)
                        from grid_master gms
                        inner join node_assignments_split nas
                        on gms.oa11 = nas.oa11
                        inner join dist_segs dsg
                        on nas.segment = dsg.segment            
                        ;
                        """, locals())

        # add cluster labels to z space scatter and sample with replacement
        # grid_array_z_cluster = pd.DataFrame(grid_master)
        # grid_array_z_cluster = pd.merge(grid_array_z_cluster, cluslabels, left_index=True, right_index=True)
        grid_array_z_cluster = np.array(grid_array_z_cluster)
        grid_array_z_cluster = grid_array_z_cluster[np.random.randint(grid_array_z_cluster.shape[0], size=10000), :]

        # 3-D scatter of sample of oa11 allocated to parent cluster
        fig = figure()
        ax = Axes3D(fig)

        for i in range(len(grid_array_z_cluster)):  # plot each point + it's index as text above
            ax.scatter(grid_array_z_cluster[i][0], grid_array_z_cluster[i][1], grid_array_z_cluster[i][2],
                       color=colour[int(grid_array_z_cluster[i][3])])

        ax.set_xlabel(Axis[0][2] + """ : """ + Axis[0][1])
        ax.set_ylabel(Axis[1][2] + """ : """ + Axis[1][1])
        ax.set_zlabel(Axis[2][2] + """ : """ + Axis[2][1])

        plt.title("""project_id = """ + project_id + """ : points in z axis space by cluster""")

        pp.savefig()
        plt.show()

        # close model outputs pdf
        pp.close()

        # output parent and child node centres and node assignments to csv
        node_assignments_split.to_csv(nasdir + """/node_assignments_split.csv""", index=False)
        cluscentres_split.to_csv(ncsdir + """/node_centres_split.csv""", index=False)

        node_assignments_split_output = psql.sqldf("""
            select   oa11
                    ,project_id
                    ,segment as initialCluster
            from node_assignments_split
            ;
                """, locals())
