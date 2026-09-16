"""
Auto-generated charts: turns the LLM's chart spec into a Plotly figure.
"""
import plotly.express as px
import pandas as pd


def build_chart(df: pd.DataFrame, spec: dict):
    chart_type = spec.get("chart_type")
    x = spec.get("x_column")
    y = spec.get("y_column")

    if x not in df.columns:
        x = None
    if y not in df.columns:
        y = None

    try:
        if chart_type == "bar" and x:
            if y:
                fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
            else:
                counts = df[x].value_counts().reset_index()
                counts.columns = [x, "count"]
                fig = px.bar(counts, x=x, y="count", title=f"Count by {x}")
        elif chart_type == "line" and x and y:
            fig = px.line(df.sort_values(x), x=x, y=y, title=f"{y} over {x}")
        elif chart_type == "scatter" and x and y:
            fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
        elif chart_type == "histogram" and x:
            fig = px.histogram(df, x=x, title=f"Distribution of {x}")
        elif chart_type == "pie" and x:
            counts = df[x].value_counts().reset_index()
            counts.columns = [x, "count"]
            fig = px.pie(counts, names=x, values="count", title=f"{x} breakdown")
        else:
            return None
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        return fig
    except Exception:
        return None
