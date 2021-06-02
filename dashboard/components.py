"""
Plotly-Dash dashboard utilities for serving a trained Sentence2Vec model and 
exploring outputs on novel user input.
"""


from dash_core_components.Markdown import Markdown
import dash_html_components as html
import dash_core_components as dcc


def header():
    return dcc.Markdown(
        children=["# Sentence-to-Vector Model"],
        id="header",
        style={
            "display": "inline-block",
            "text-align": "center",
            "width": "100%",
            "padding": "10px",
            "margin": "10px"
        }
    )


def input_form(width="100%", height="100%"):
    return html.Div(
        children=[
            #dcc.Markdown("##### Submit Input"),
            dcc.Input(
                placeholder="Ask a question about diabetes...", 
                id="input_form_input", 
                style={"width": "70%"}
            ),
            html.Button(
                children=["Submit"], 
                id="input_form_button",
                style={"width": "30%"}
            ),
            html.Div(
                children=[],
                id="input_form_output",
                style={"visibility": "hidden"}
            )
        ],
        id="input_form",
        style={
            "display": "inline-block",
            "text-align": "center",
            "padding": "10px",
            "margin": "10px",
            "width": width,
            "height": height,
            #"border": "2px solid",
            #"border-radius": "5px"
        }
    )


def projected_manifold_plot(
        figure, width="100%", height="100%", color="#ccebff"):
    # override figure layout properties
    figure.update_layout(
        height=600,
        showlegend=False
    )

    figure.layout.plot_bgcolor = color
    figure.layout.paper_bgcolor = color

    return html.Div(
        children=[
            dcc.Markdown("##### Projected Question Vector Manifold"),
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
        style={
            "display": "inline-block",
            "text-align": "center",
            "padding": "10px",
            "margin": "10px",
            "width": width,
            "height": height,
            "border": "2px solid",
            "border-radius": "5px",
            "border-color": color,
            "background-color": color
        }
    )


def summary_section(width="100%", height="100%"):
    return html.Div(
        children=[
            dcc.Markdown(
                children=[f"""
                    ##### Input Summary
                """]
            )
        ],
        id="summary_section",
        style={
            "display": "inline-block",
            "text-align": "center",
            "padding": "10px",
            "margin": "10px",
            "width": width,
            "height": height,
            #"border": "2px solid",
            #"border-radius": "5px"
        }
    )