from setup import *
project_id = 'P1'

adir = path + '/data/axis/' + project_id
oadir = path + '/data/spines/oa11'
cdir = path + '/data/clustering/' + project_id

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

# assign each oa11 to nearest initial node (nearest neaighbour lookup)
nearest_node = np.zeros((len(grid_array_z), 4))
nearest_node[:, :-1] = grid_array_z

for row in range(len(nearest_node)):
    pt = nearest_node[row, [0, 1, 2]]
    dist, ind = spatial.KDTree(init).query(pt)
    nearest_node[row, 3] = ind

# frequency of initial node allocations
plt.hist(nearest_node[:, 3], bins=n_clusters)
plt.title("""project_id = """ + project_id + """ : nearest node assignments : cluster nodes = """ + str(n_clusters),
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
entropy

# plot node frequencies
fig = plt.figure()
plt.hist(nodes, bins=n_clusters)
plt.title("""project_id = """ + project_id + """ : cluster nodes = """ + str(n_clusters) + """ : iterations = """ + str(
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
plt.title("""project_id = """ + project_id + """ : "2-D multidimensional scaling of 3-D axis : clusters = """ + str(
    n_clusters) + """ : iterations = """ + str(max_iter) + """ : entropy = """ + str(int(entropy)), fontsize=8)
plt.xlabel('reduced dimension 1')
plt.ylabel('reduced dimension 2')
pp.savefig()

# cluster centres dataframe
cluscentres = pd.DataFrame(cluscentres)
cluscentres

# get cluster centres next to initialisation nodes
initialisations = pd.DataFrame(init)
cluscentres = pd.merge(initialisations, cluscentres, left_index=True, right_index=True)
cluscentres

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
cluslabels

# join oa11 to nodes assignent on index
np.testing.assert_array_equal(oa11['oa11'].count(), cluslabels['initialCluster'].count(),
                              nearest_node['nearestnode'].count())
node_assignments = pd.merge(oa11, cluslabels, left_index=True, right_index=True)
node_assignments = pd.merge(node_assignments, nearest_node, left_index=True, right_index=True)
node_assignments['project_id'] = project_id
node_assignments

# node frequencies
node_frequencies = pd.DataFrame(node_assignments.groupby(['initialCluster'])['oa11'].count())

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

# output node assignments and axis
cluscentres['cluster'] = cluscentres.index
node_assignments_nodes = pysqldf("""
    select   nas.oa11
            ,nas.initialCluster
            ,cls.[0_y] as axis0
            ,cls.[1_y] as axis1
            ,cls.[2_y] as axis2
    from node_assignments nas
    inner join cluscentres cls
    on nas.initialCluster = cls.cluster
    ;
    """)

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

plt.title("""project_id = """ + project_id + """ : initial nodes""")

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

plt.title("""project_id = """ + project_id + """ : cluster nodes = """ + str(n_clusters) + """ : iterations = """ + str(
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

plt.title("""project_id = """ + project_id + """ : points in z axis space by cluster""")

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

plt.title("""project_id = """ + project_id + """ : points in z axis space by cluster""")

pp.savefig()

pp.close()

