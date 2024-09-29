import pandas as pd
from flask import Flask, make_response
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import os
import base64

app = Flask(__name__)
dash_app = Dash(__name__, server=app, external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css.bootstrap.min.css"])

def read_data():
    """Read the data from the CSV files and return it as pandas DataFrames."""
    exosuits_data = pd.read_csv("csv/exosuits.csv")
    tech_data = pd.read_csv("csv/technologies.csv", sep=";")
    return exosuits_data, tech_data

def process_data(exosuits_data, tech_data):
    """Process the data to prepare it for visualization."""
    tech_info = tech_data.set_index("TID")[["TRL", "PriceAvg"]].to_dict()

    tech_columns = [f"T{i}" for i in range(1, 125)]

    avg_trl_list = []
    total_price_list = []
    pids = []

    for pid, row in exosuits_data.iterrows():
        tech_trls = []
        tech_prices = []

        for i, col in enumerate(tech_columns, start=1):
            if row[col] == 1:
                tech_trls.append(tech_info["TRL"].get(i, 0))
                tech_prices.append(tech_info["PriceAvg"].get(i, 0))

        avg_trl = sum(tech_trls) / len(tech_trls) if tech_trls else 0
        total_price = sum(tech_prices)

        avg_trl_list.append(avg_trl)
        total_price_list.append(total_price)
        pids.append(pid + 1)

    df = pd.DataFrame({
        "PID": pids,
        "Average_TRL": avg_trl_list,
        "Total_Price": total_price_list,
    })

    return df

def create_scatter_plot(df):
    """Creates a Plotly scatter plot object."""
    fig = go.Figure(data=[
        go.Scatter(
            x=df["Average_TRL"],
            y=df["Total_Price"],
            mode="markers",
            text=[f"PID: {pid}" for pid in df["PID"]],
            customdata=df["PID"],
            hovertemplate=(
                "<b>%{text}</b><br>Average TRL: %{x}<br>Total Price: %{y}<br>"
                "<extra></extra>"
            )
        )
    ])

    fig.update_layout(
        title="Exosuits: Total Price vs. Average TRL",
        xaxis_title="Average TRL of Exosuit Technologies",
        yaxis_title="Total Price of Exosuit Technologies ($)",
        showlegend=False,
    )

    return fig

@app.route("/image/<int:pid>")
def serve_image(pid):
    """Serves the image corresponding to the PID as a base64-encoded string."""
    img_path = os.path.join("csv/images", f"exoskeleton ({pid}).jpg")

    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
            response = make_response(f"<img src='data:image/jpeg;base64,{img_b64}' width='320' height='480'>")
            response.headers["Content-Type"] = "text/html"
            return response

    return make_response("")

def serve_dashboard():
    """Serves the dashboard interface."""
    exosuits_data, tech_data = read_data()
    df = process_data(exosuits_data, tech_data)
    fig = create_scatter_plot(df)

    dash_app.layout = html.Div([
        html.Div([
            dcc.Graph(
                id="scatter-plot",
                figure=fig,
            ),
            html.Div(id="hover-output", style={"width": "320px", "height": "480px", "float": "right"}),
        ], style={"display": "flex"}),
    ])

    dash_app.run_server()

@dash_app.callback(
    Output("hover-output", "children"),
    [Input("scatter-plot", "hoverData")]
)
def update_image(hover_data):
    """Updates the image in response to hover data."""
    if hover_data:
        pid = hover_data["points"][0]["customdata"]
        img_url = f"/image/{pid}"
        img_tag = html.Div([html.Iframe(src=img_url, width="320", height="480")])
        return img_tag

    return ""

if __name__ == "__main__":
    serve_dashboard()
