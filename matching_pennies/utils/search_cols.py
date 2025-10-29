import polars as pl 

def search_cols(search_term: str, df:pl.DataFrame):
    columns = df.columns
    return [col for col in columns if search_term.lower() in col.lower()]