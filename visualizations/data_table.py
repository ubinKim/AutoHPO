"""
Create a table with the mean values of the data in the column of DataFrame.
"""

import pandas as pd
import plotly.graph_objects as go


# Round to 4 decimal places and remove trailing zeros if exist
def round_float(num):
    rounded_num = "{:.4f}".format(num)  # Alternative: str(round(num, 4))
    return rounded_num.rstrip('0').rstrip('.') if '.' in str(rounded_num) else str(rounded_num)


result_file = pd.read_csv('../results/non_optimized.csv')

fig = go.Figure()
header = ['lr', 'dropout', 'hidden_unit', 'tr_acc', 'val_acc', 'elapsed_time']
values = []

for col in header:
    col_mean = result_file[col].mean()
    rounded_mean = round_float(col_mean)
    values.append([float(rounded_mean)])

# Create table
fig.add_trace(go.Table(
    header=dict(values=result_file.columns),
    cells=dict(values=values)))

fig.show()

# --- [Optional] Change the table style --- #
# fig.add_trace(go.Table(
#     header=dict(values=df.columns,
#                 line_color='white',
#                 fill_color='#E3F2C1'
#                 align="center"),
#     cells=dict(values=values,
#                line_color='white',
#                fill_color='#F6FFDE'
#                align="center")))
# ------------------------------------------ #
