"""
Sentence-to-Vec Dash app for model serving and visualization.
"""


from sklearn import cluster
import yaml
import torch
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from dashboard.components import header, input_form, projected_manifold_plot, summary_section

from dataset.qa_triplet_dataset import QATripletDataset
from model.max_over_time_cnn import MaxOverTimeCNN

from scraper.utils.text_utils import process_text


def main():
    """ Model Setup """

    with open('config/app_cfg.yml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    
    dataset = QATripletDataset(
            config['database_file'], config['database_table'], 
            config['vocab_file'])

    dataloader = dataset.build_dataloader(
        batch_size=config['batch_size'], shuffle=True)

    encoder = MaxOverTimeCNN(
        len(dataset.get_vocab()), config['wordvec_dim'], 
        config['sentvec_dim'], acceleration=True)

    encoder.load_state_dict(torch.load(
        'artifacts/models/max_over_time_cnn.pt', 
        map_location=torch.device('cpu')))

    encoder.eval()

    # grab a batch and encode anchor questions to vectors
    batch = next(iter(dataloader))
    question_str = batch['anc_str']
    question_enc = encoder(batch['anc_idxs'])

    # detach from pytorch and convert to numpy and dictionary
    question_enc = question_enc.detach().numpy()
    question_enc_dict = {
        i: list(row) for i, row in enumerate(question_enc)}

    # cluster vectors
    clustering = DBSCAN(eps=0.006, min_samples=10, n_jobs=-1, metric='cosine')
    labels = clustering.fit_predict(question_enc)
    # optics = OPTICS(min_samples=10, metric='cosine')
    # optics.fit(question_enc)
    # labels = optics.labels_

    num_clusters = len(np.unique(labels))
    perc_labeled = 100 * (len(labels[labels != -1]) / len(labels))
    print(f'[Num. Clusters]: {num_clusters}, [Perc. Non-Outliers]: {perc_labeled:.1f}')

    # project sentence vector manifold
    tsne = TSNE(
        n_components=3, metric='cosine', random_state=42)
    question_proj = tsne.fit_transform(question_enc)

    df = pd.DataFrame({
        'sentence': question_str,
        'vector': [x for x in question_enc],
        'label': labels,
        'pc_1': question_proj[:, 0],
        'pc_2': question_proj[:, 1],
        'pc_3': question_proj[:, 2],
    })

    fig = px.scatter_3d(
        df, x='pc_1', y='pc_2', z='pc_3', color='pc_3', 
        hover_data=['sentence'])
    fig.update_traces(marker_size=5)
    fig.layout.coloraxis.showscale = False
    fig.update_layout(showlegend=False)

    # nearest neighbors model
    neigh = NearestNeighbors(n_neighbors=6, metric='cosine')

    """ App Setup """

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        header(),
        html.Div(
            children=[
                html.Div(
                    children=[
                        input_form(
                            style={
                                "vertical-align": "top",
                                "justify-content": "center",
                                "padding": "10px",
                                "margin": "10px",
                                "width": "93%",
                                "height": "100%",
                                "border": "2px solid",
                                "border-radius": "5px",
                                "border-color": "#d4d6d6",
                                "background-color": "#d4d6d6"
                            }
                        ),
                        summary_section(
                            style={
                                "justify-content": "center",
                                "padding": "10px",
                                "margin": "10px",
                                "width": "93%",
                                "height": "100%",
                                "border": "2px solid",
                                "border-radius": "5px",
                                "border-color": "#d4d6d6",
                                "background-color": "#d4d6d6"
                            }
                        )
                    ],
                    style={
                        "vertical-align": "top",
                        "display": "inline-block",
                        "width": "35%"
                    }
                ),
                projected_manifold_plot(
                    fig,
                    style={
                        "display": "inline-block",
                        "vertical-align": "top",
                        "justify-content": "center",
                        "padding": "10px",
                        "margin": "10px",
                        "width": "55%",
                        "height": "100%",
                        "border": "2px solid",
                        "border-radius": "5px",
                        "border-color": "#d4d6d6",
                        "background-color": "#d4d6d6"
                    }
                )
            ],
            id="body",
            style={
                "text-align": "center",
                "padding": "10px",
                "margin": "10px",
                "width": "97%",
                "height": "100%",
                "border": "2px solid",
                "border-radius": "10px",
                "border-color": "#c9c9c9",
                "background-color": "#c9c9c9"
            }
        ),
        dcc.Store(
            id="local_question_enc", 
            data=question_enc_dict
        )
    ])


    @app.callback(
        Output("raw_input", "children"),
        [Input("input_form_button", "n_clicks"),
        State("input_form_input", "value")]
    )
    def update_raw_input(n_clicks, value):
        if n_clicks > 0:
            return value


    @app.callback(
        Output("processed_input", "children"),
        [Input("input_form_button", "n_clicks"),
        State("input_form_input", "value")]
    )
    def update_processed_input(n_clicks, value):
        if n_clicks > 0:
            return process_text(value)


    @app.callback(
        Output("nearest_neighbors", "children"),
        [Input("input_form_button", "n_clicks"),
        State("input_form_input", "value"),
        State("local_question_enc", "data")]
    )
    def update_nearest_neighbors(n_clicks, value, question_enc):
        if n_clicks > 0:
            # encode user question to sentence representation
            proc_value = process_text(value)
            value_idx = dataset.str_to_idx_tensor(proc_value)
            value_enc = encoder(
                value_idx + [torch.LongTensor([0])])[0]

            question_enc_np = np.concatenate(
                [np.expand_dims(np.array(vec), axis=0) \
                for _, vec in question_enc.items()], axis=0)

            # append to question_enc array
            question_enc_np = np.append(
                question_enc_np, np.expand_dims(
                    value_enc.detach().numpy(), axis=0), axis=0)

            # compute nearest neighbors
            nbrs = neigh.fit(question_enc_np)
            _, nn_idx = nbrs.kneighbors(question_enc_np)

            # get nearest neighbors on last sample
            neighbors = [question_str[i] for i in nn_idx[-1, 1:]]
            
            neighbor_divs = [
                html.Div(
                    f"{i+1}. {neighbor}", 
                    style={"text-align": "left"}) \
                for i, neighbor in enumerate(neighbors)]

            return neighbor_divs


    @app.callback(
        Output("nearest_cluster", "children"),
        [Input("input_form_button", "n_clicks"),
        State("input_form_input", "value"),
        State("local_question_enc", "data")]
    )
    def update_nearest_cluster(n_clicks, value, question_enc):
        if n_clicks > 0:
            # encode user question to sentence representation
            proc_value = process_text(value)
            value_idx = dataset.str_to_idx_tensor(proc_value)
            value_enc = encoder(
                value_idx + [torch.LongTensor([0])])[0]

            # convert question_enc dict back to numpy array
            question_enc_np = np.concatenate(
                [np.expand_dims(np.array(vec), axis=0) \
                for _, vec in question_enc.items()], axis=0)

            # append to question_enc array
            question_enc_np = np.append(
                question_enc_np, np.expand_dims(
                    value_enc.detach().numpy(), axis=0), axis=0)
            
            # cluster vectors
            clustering = DBSCAN(eps=0.006, min_samples=10, n_jobs=-1, metric='cosine')
            labels = clustering.fit_predict(question_enc_np)
            # optics = OPTICS(min_samples=10, metric='cosine')
            # optics.fit(question_enc)
            # labels = optics.labels_

            df = pd.DataFrame({
                'sentence': question_str + [proc_value],
                'vector': [x for x in question_enc_np],
                'label': labels
            })

            # get list standard stop words
            stop_words = stopwords.words('english')

            # initialize word stemmer
            stemmer = nltk.stem.PorterStemmer()

            # stem stop words
            stop_words = [stemmer.stem(word) for word in stop_words]

            # add a summary column
            df['summary'] = None

            # add column for average cluster vector
            df['avg_vector'] = None

            # infer cluster topics
            for label in sorted(df['label'].unique()):
                # get all samples with this label
                samples = df[df['label'] == label]['sentence']
                
                # get all vectors with this label
                vectors = np.stack(
                    df[df['label'] == label]['vector'].to_numpy())
                
                # compute average vec for this cluster label
                avg_vector = np.mean(vectors, axis=0)
                
                # add average vector to dataframe
                df.loc[df['label'] == label, 'avg_vector'] = \
                    df.loc[df['label'] == label, 'avg_vector'].map(
                        lambda x: avg_vector)
                
                # convert samples to a list
                samples = samples.tolist()
                
                # tokenize samples by whitespace
                tokens = [[word for word in sentence.split(' ')] 
                    for sentence in samples]
                
                # flatten samples list
                tokens = [inner for outer in tokens for inner in outer]
                
                # stem tokens
                tokens = [stemmer.stem(token) for token in tokens]
                
                # filter stopwords
                tokens = [token for token in tokens if not token in stop_words]
                
                # get token frequencies
                fdist = nltk.FreqDist(tokens)
                
                # get summary string from 5 most frequent tokens
                summary = ' '.join(
                    [token for token, _ in fdist.most_common(5)])
                
                # add summary string to dataframe
                df.loc[df['label'] == label, 'summary'] = summary

            # get list of cluster labels
            cluster_summaries = df['summary'].drop_duplicates().tolist()

            # create numpy array of average cluster vectors
            cluster_vecs = np.stack(
                df['avg_vector'].drop_duplicates().tolist(), axis=0)

            # compute closest avg cluster vector to this query vector
            dists = np.linalg.norm(
                cluster_vecs - value_enc.detach().numpy(), ord=1, axis=1)

            value_cluster = cluster_summaries[np.argmin(dists)]

            return value_cluster
    

    app.run_server(debug=False)


if __name__ == '__main__':
    main()
