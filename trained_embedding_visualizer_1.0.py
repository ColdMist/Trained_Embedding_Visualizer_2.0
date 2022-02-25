import pandas as pd
import numpy as np
import itertools
from sklearn.cluster import KMeans
import pprint
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from adjustText import adjust_text
from sklearn.manifold import TSNE


def original_index_to_type(data, value_to_converted):

    converted = []
    data = dict(zip(data[0], data[1]))

    for d in value_to_converted:
        converted.append(data[str(d)])
    return np.array(converted)

def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        print('does not contain duplicates')
    else:
        print('contain duplicates')

def prepare(type_A):
    a = []
    for i in type_A[0]:
        a.append(i)
    return np.array(a)

def entity_to_id_conversion(entities):
    id_converted = []
    ###Should be changed for both types 0 1 for sem
    entity_to_id_dict = dict(zip(entity_to_id[1], entity_to_id[0]))

    for entity in entities:
        id_converted.append(entity_to_id_dict[entity])
    return np.array(id_converted)

def scatter_text(x, y, text_column, hue ,data, title, xlabel, ylabel):
    """Scatter plot with country codes on the x y coordinates
       Based on this answer: https://stackoverflow.com/a/54789170/2641825"""
    # Create the scatter plot
    p1 = sns.scatterplot(x, y, data=data, hue=hue)
    # texts = []
    # # Add text besides each point
    # for line in range(0,data.shape[0]):
    #      # p1.text(data[x][line]+0.01, data[y][line],
    #      #         data[text_column][line], horizontalalignment='left',
    #      #         size='small', color='black', weight='semibold')
    #      texts.append(p1.text(data[x][line],data[y][line],data[text_column][line]))
    # Set title and axis labels
    #adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1

#Read trained embeddings
entity_embedding = np.load('/home/mirza/visualizations/trained_embeddings_complex_semantic/entity_embedding.npy')
relation_embedding = np.load('/home/mirza/visualizations/trained_embeddings_complex_semantic/relation_embedding.npy')

#Read the dictionary for original text
entity_to_id = pd.read_table('/home/mirza/visualizations/FB15k_semantic_matching/entities.dict', header=None)
#entity_to_id= entity_to_id[[1, 0]]

#entity_to_id = pd.DataFrame(entity_to_id.T)

relation_to_id = pd.read_table('/home/mirza/visualizations/FB15k_semantic_matching/relations.dict', header=None)
#relation_to_id= relation_to_id[[1, 0]]

entity_embedding = pd.DataFrame(entity_embedding)
stats = pd.read_pickle('/home/mirza/visualizations/FB15k/train_stats.pkl')

#Best indicated types
#'/education/educational_institution', '/book/author', '/film/film'

#indicated_types = ['/education/educational_institution', '/book/author', '/film/film', '/location/location']
#indicated_types = ['/education/educational_institution', '/book/author' ,'/film/film', '/tv/tv_actor',  '/tv/tv_program', '/music/instrument', '/travel/travel_destination', '/business/employer']
indicated_types = ['/film/film'  , '/tv/tv_program', '/music/instrument',
                   '/organization/endowed_organization'
                   ,'/location/statistical_region', '/music/group_member']
ids = []
types_with_ids = []

for i in range(len(indicated_types)):
    type_ = np.array(stats.loc[stats['relation_name'] == indicated_types[i]]['matched_entities'])
    type_prepared = prepare(type_)
    type_ids = entity_to_id_conversion(type_prepared)
    type_ids_type = np.repeat(indicated_types[i], len(type_ids))
    type_combined = np.c_[type_ids, type_ids_type]
    ids.append(type_ids)
    types_with_ids.append(type_combined)

ids = tuple(ids)
ids = (list(np.concatenate(ids)))
#total_ids = set(list(np.concatenate(ids)))
total_ids = np.unique(ids)
com = np.vstack(np.array(types_with_ids))



#total_ids = (list(np.concatenate(args)))
#total_ids = unique(total_ids)
#com = np.vstack( [type_A_combined , type_B_combined, type_C_combined, type_D_combined, type_E_combined, type_F_combined] )

checkIfDuplicates_1(total_ids)

dim_reduced = entity_embedding.loc[total_ids]
original_index = dim_reduced.index

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
tsne_result = tsne.fit_transform(dim_reduced)

tsne_df = pd.DataFrame()

tsne_df['tsne-one'] = tsne_result[:,0]
tsne_df['tsne-two'] = tsne_result[:,1]
#tsne_df['tsne-three'] = tsne_result[:,2]
tsne_df['original_index'] = original_index

type = original_index_to_type(pd.DataFrame(com), tsne_df['original_index'])
tsne_df['type'] =type


# import plotly.express as px
#
# fig = px.scatter_3d(tsne_df, x='tsne-one', y='tsne-two', z='tsne-three',
#               color='type')
# fig.show()
c_palate = {}
colors = ['tab:purple','tab:orange','tab:green','tab:red','tab:blue','tab:brown']
for i,v in zip(indicated_types, colors):
    #print(indicated_types[i])
    c_palate[i] = v


plt.figure(figsize=(16,16))
plot = sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue='type',
    data=tsne_df,
    palette=c_palate,
    legend="full",
    alpha=0.8
)
plt.setp(plot.get_legend().get_texts(), fontsize='15')
plt.show()