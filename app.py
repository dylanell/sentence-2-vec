"""
Sentence-to-Vec Dash app for model serving and visualization.
"""


import yaml
import torch
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.manifold import TSNE
import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State

from dashboard.components import header, input_form, projected_manifold_plot, summary_section

from datasets.qa_triplet_dataset import QATripletDataset
from models.max_over_time_cnn import MaxOverTimeCNN


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
        len(dataset.get_vocab()), config['wordvec_dim'], config['sentvec_dim'], 
        acceleration=True)

    encoder.load_state_dict(torch.load(
        'artifacts/models/max_over_time_cnn.pt', 
        map_location=torch.device('cpu')))

    encoder.eval()

    # grab a batch and encode anchor questions to vectors
    batch = next(iter(dataloader))
    question_str = batch['anc_str']
    question_enc = encoder(batch['anc_idxs'])

    # detach from pytorch and convert to numpy
    question_enc = question_enc.detach().numpy()

    clustering = DBSCAN(eps=0.006, min_samples=10, n_jobs=-1, metric='cosine')
    labels = clustering.fit_predict(question_enc)

    # optics = OPTICS(min_samples=10, metric='cosine')
    # optics.fit(question_enc)
    # labels = optics.labels_

    num_clusters = len(np.unique(labels))
    perc_labeled = 100 * (len(labels[labels != -1]) / len(labels))
    print(f'[Num. Clusters]: {num_clusters}, [Perc. Non-Outliers]: {perc_labeled:.1f}')

    tsne = TSNE(n_components=3, metric='cosine')
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

    """ App Setup """

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        header(),
        html.Div(
            children=[
                input_form(width="20%"),
                projected_manifold_plot(fig, width="50%"),
                summary_section(width="20%")
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
                "border-color": "#e6f5ff",
                "background-color": "#e6f5ff"
            }
        )
    ])


    @app.callback(
        Output("input_form_output", "children"),
        [Input("input_form_button", "n_clicks"),
        State("input_form_input", "value")])
    def submit_input(n_clicks, value):
        if n_clicks == 0:
            return None
        else:
            return value


    app.run_server(debug=True)


if __name__ == '__main__':
    main()
