import dash_bio
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/dash-bio-docs-files/master/manhattan_data.csv')


dash_bio.ManhattanPlot(
    dataframe=df,
)