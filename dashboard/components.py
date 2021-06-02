"""
Plotly-Dash dashboard utilities for serving a trained Sentence2Vec model and 
exploring outputs on novel user input.
"""


from dash_core_components.Markdown import Markdown
import dash_html_components as html
import dash_core_components as dcc


def header():
    return html.H1(
        children=[html.B("Sentence-to-Vector Model")],
        id="header",
        style={
            "display": "inline-block",
            "text-align": "center",
            "width": "100%",
            "padding": "10px",
            "margin": "10px"
        }
    )


def input_form(style=None):
    if style is None:
        style = style={
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
    
    return html.Div(
        children=[
            dcc.Input(
                placeholder="Ask a question about diabetes...", 
                id="input_form_input", 
                style={"width": "70%"}
            ),
            html.Button(
                children=["Submit"], 
                id="input_form_button",
                style={"width": "30%"},
                n_clicks=0
            )
        ],
        id="input_form",
        style=style
    )


def projected_manifold_plot(figure, style=None):
    if style is None:
        style = {
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
    
    # override figure layout properties
    figure.update_layout(
        height=600,
        showlegend=False
    )

    figure.layout.plot_bgcolor = style["background-color"]
    figure.layout.paper_bgcolor = style["background-color"]

    return html.Div(
        children=[
            html.H5(html.B("Projected Question-Vector Manifold")),
            dcc.Graph(
                figure=figure, 
                style={
                    "width": "100%", 
                    "height": "100%",
                    "text-align": "center"
                }
            )
        ],
        id="projected_manifold_plot",
        style=style
    )


def summary_section(style=None):
    if style is None:
        style = {
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
    
    return html.Div(
        children=[
            html.H5(html.B("Input Summary")),
            html.Div(html.B("Raw Input:"), style={"text-align": "left"}),
            html.Div(id="raw_input", style={"text-align": "left"}),
            html.Div(
                html.B("Processed Input:"), style={"text-align": "left"}),
            html.Div(id="processed_input", style={"text-align": "left"}),
            html.Div(
                html.B("Five Nearest Neighbors:"), style={"text-align": "left"}),
            html.Div(id="nearest_neighbors", style={"text-align": "left"}),
            html.Div(
                html.B("Nearest Cluster:"), style={"text-align": "left"}),
            html.Div(id="nearest_cluster", style={"text-align": "left"})
        ],
        id="summary_section",
        style=style
    )