import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
import seaborn as sns
from scipy.spatial import ConvexHull
from adjustText import adjust_text
import torch


def draw_convex_haul(dfs, palette):
    i = 0
    for df in dfs:
        points = df[['x-axis', 'y-axis']].values
        convex_hull = ConvexHull(points)
        x_hull = np.append(points[convex_hull.vertices, 0],
                           points[convex_hull.vertices, 0][0])
        y_hull = np.append(points[convex_hull.vertices, 1],
                           points[convex_hull.vertices, 1][0])
        plt.fill(x_hull, y_hull, alpha=0.3, c=palette[i])
        i+=1

def fetch_random_indexes_per_cluster(dfs, number_of_samples_per_df = 100):
    random_indexes_overall = []
    for df in dfs:
        random_indexes_per_df = np.random.choice(list(df.index), size=number_of_samples_per_df, replace=False)
        #print(random_indexes_per_df)
        random_indexes_overall.extend(random_indexes_per_df)
    #exit()
    return  random_indexes_overall

def annotate_members(df, random_indexes_to_annotate, palette, x_axis = 'x-axis', y_axis = 'y-axis', text_column = 'text'):
    overal_texts = []
    for i in random_indexes_to_annotate:
        # plt.text(df[x_axis][i], df[y_axis][i], df[text_column][i], horizontalalignment='left', size='medium',
        #      color=palette[df['label'][i]], weight='semibold')
    #     plt.text(df[x_axis][i], df[y_axis][i], df[text_column][i], horizontalalignment='left', size='medium',
    #              color='black', weight='semibold')
        overal_texts.append(plt.text(df[x_axis][i], df[y_axis][i], df[text_column][i]))
    #print(overal_texts)
    adjust_text(overal_texts)
    #adjust_text(overal_texts, arrowprops=dict(arrowstyle='->', color='red'))
    #return overal_texts


def annotate_centroids(centroids, palette):
    i = 1
    j = 0
    for centroid in centroids:
        text = 'centroid ' + str(i)
        plt.text(centroid[0], centroid[1], 'x' , horizontalalignment='left', size=20,
             color=palette[j], weight='bold')
        i+=1
        j+=1

def reduce_dimension(alg_type = 'pca', dim = 2):
    init_var = None,
    if alg_type == 'pca':
        init_var = PCA(dim)
    elif alg_type == 'TSNE':
        init_var = TSNE(n_components=dim, learning_rate='auto', init = 'random')
    elif alg_type == 'ISOMAP':
        init_var = Isomap(n_components=dim)
    elif alg_type == 'SpectralEmbedding':
        init_var = SpectralEmbedding(n_components=dim)
    else:
        print('dimensionality reduction technique is not listed')
        exit()
    return init_var


def scatter_plot(x, y, data, hue, n_clusters, palette ,title, xlabel, ylabel):
    palette_subset = [palette[i] for i in range(n_clusters)]
    p1 = sns.scatterplot(x, y, data=data, hue=hue, palette=palette_subset )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1

if __name__ == "__main__":
    #TODO input the embeddings, entity to id file
    trained_emb_dir = '/home/mirza/PycharmProjects/pythonProject1/Negative-Sampling_LM/SANS-master-KMeans/data/umls/trained_embedding/UMLS_Embedings-transformer.npy'
    #trained_emb_2_dir = '/home/mirza/PycharmProjects/pythonProject1/Negative-Sampling_LM/SANS-master-KMeans/data/umls/trained_embedding/sans-kmeans-pretrained (1).npy'
    entity_to_id_dict_dir = '/home/mirza/PycharmProjects/pythonProject1/Negative-Sampling_LM/SANS-master-KMeans/data/umls/entities.dict'
    #TODO input the embedding type: 1. pca, 2. TSNE, 3. ISOMAP 4. SpectralEmbedding
    alg_type = 'ISOMAP'
    #TODO input the number of text samples in the graph per cluster
    n_text = 50
    #TODO input the number of clusters
    n_clusters = 8
    #TODO input the color pallete
    palette = sns.color_palette("tab10")

    #TODO if we want to use same random numbers!
    np.random.seed(42)

    # trained_Emb = torch.load('/home/mirza/PycharmProjects/pythonProject1/Negative-Sampling_LM/SANS-master-KMeans/data/mahfuza apu data/MTE_FS_entity.pkl', map_location=torch.device('cpu'))
    # trained_Emb = pd.DataFrame(trained_Emb.weight.detach().numpy())

    trained_Emb=np.load(trained_emb_dir)
    entity_to_id = pd.read_table(entity_to_id_dict_dir, header=None)
    entity_to_id_text = entity_to_id[1].values


    #Reduce reduce the embeddings into lower dimension, currently only 2d visualization is possible
    trained_Emb_low_d = reduce_dimension(alg_type=alg_type, dim=2).fit_transform(trained_Emb)

    #initialize kmeans
    kmeans = KMeans(n_clusters=n_clusters,init='k-means++', n_init=10, max_iter=1000, verbose=0, random_state=1234)

    # predict the labels of clusters.
    label = kmeans.fit_predict(trained_Emb_low_d)
    centroids = kmeans.cluster_centers_

    unique_label = np.unique(label)
    info_array = np.c_[trained_Emb_low_d,label,entity_to_id_text]

    info_df = pd.DataFrame(info_array)



    info_df.columns = ['x-axis', 'y-axis', 'label', 'text']

    for i in unique_label:
        class_i_member_size = len(info_df.loc[info_df['label']==i])
        print(i, class_i_member_size)

    individual_df_per_cluster = [info_df.loc[info_df['label']==i] for i in unique_label]
    #random_indexes_to_annotate = fetch_random_indexes_per_cluster(individual_df_per_cluster, number_of_samples_per_df=n_text)
    random_indexes_to_annotate = np.random.choice(list(info_df.index), size=n_text, replace=False)
    print(random_indexes_to_annotate)

    plt.figure(figsize=(20,10))

    scatter_plot('x-axis', 'y-axis',
                 data = info_df,
                 hue = 'label',
                 n_clusters=n_clusters,
                 palette = palette,
                 title = alg_type,
                 xlabel = 'x-axis',
                 ylabel = 'y-axis')

    overal_texts = annotate_members(info_df, random_indexes_to_annotate, palette, 'x-axis', 'y-axis', 'text')
    annotate_centroids(centroids, palette)
    draw_convex_haul(dfs=individual_df_per_cluster, palette=palette)

    plt.legend()
    plt.show()

