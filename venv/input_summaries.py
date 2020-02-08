from setup import *


class InputSummaries:

    def __init__(self, project_id):
        """produce summaries of a set of input features provided"""

        self.project_id = project_id

        dir = path + '/data/inputs/' + self.project_id  # features input file path
        odir = path + '/data/input_summaries/' + self.project_id  # input features summary output path
        pp = PdfPages(odir + '/level_summaries.pdf')  # open input features summary pdf

        lm = pd.DataFrame()
        sm = pd.DataFrame()

        for file in os.listdir(dir):

            # for each input feature count rows, number of oa's and number of levels
            feature = file[:-4]
            df = pd.read_csv(dir + '/' + str(file))
            rc = df['oa11'].count()
            oc = df['oa11'].nunique()
            lc = df['level'].nunique()
            ss = pd.DataFrame({'feature': feature, 'row_count': rc, 'oa11_count': oc, 'levels_count': lc}, index=[0])
            sm = sm.append(ss)

            # for each input feature, for each level, count rows, sum/min/max metric
            ls = (df.groupby(['level']).agg(count=('metric', 'count'),sum=('metric', 'sum'),min=('metric', 'min'),max=('metric', 'max'))).reset_index()
            ls['feature'] = feature
            lm = lm.append(ls)

            # for each feature plot distribution of sum of metric
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
