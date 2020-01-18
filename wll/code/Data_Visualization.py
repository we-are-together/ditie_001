import pandas as pd
import  plotly.graph_objects as go
class Csv_View:
    def view_bar(self):
        filename=input('请输入CSV文件名:')
        df = pd.read_csv('{}.csv'.format(filename)) 
        gp = df.groupby(["ItemName"])["ItemName"].count()
        data = dict(gp)
        name = list(data.keys())
        count = list(data.values())
        line1 = go.Bar(x=name, y=count)
        fig = go.Figure(line1)
        fig.show()
    def view_pie(self):
        filename = input('请输入CSV文件名:')
        df = pd.read_csv('{}.csv'.format(filename))
        gp = df.groupby(["ItemName"])["ItemName"].count()
        data = dict(gp)
        name = list(data.keys())
        count = list(data.values())
        fig = go.Figure(data=[go.Pie(labels=name, values=count)])
        fig.show()
V=Csv_View()
V.view_bar()


