import pandas as pd
import plotly.graph_objects as go
import os

def read_data():
    """Read the data from the CSV files and return it as pandas DataFrames."""
    exosuits_data = pdread_csv("csv/exosuits.csv")
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

    # Create a DataFrame with all necessary information
    df = pd.DataFrame({
        "PID": pids,
        "Average_TRL": avg_trl_list,
        "Total_Price": total_price_list,
        "Image_Path": [os.path.abspath(os.path.join("csv/images/exoskeleton(1).jpg")).replace("\\", "/") for pid in pids]
    })

    return df

def create_interactive_plot(df):
    """Create and display an interactive scatter plot."""
    fig = go.Figure(data=[
        go.Scatter(
            x=df["Average_TRL"],
            y=df["Total_Price"],
            mode="markers",
            text=[f"PID: {pid}" for pid in df["PID"]],
            customdata=df["Image_Path"],
            hovertemplate=(
                "<b>%{text}</b><br>Average TRL: %{x}<br>Total Price: %{y}"
                "<br><img src='file://%{customdata}' width='150' height='150'><extra></extra>"
            )
        )
    ])

    fig.update_layout(
        title="Exosuits: Total Price vs. Average TRL",
        xaxis_title="Average TRL of Exosuit Technologies",
        yaxis_title="Total Price of Exosuit Technologies ($)",
        showlegend=False
    )

    fig.show()

def main():
    """Main function to run the data reading, processing, and visualization."""
    exosuits_data, tech_data = read_data()
    df = process_data(exosuits_data, tech_data)
    create_interactive_plot(df)

if __name__ == "__main__":
    main()
