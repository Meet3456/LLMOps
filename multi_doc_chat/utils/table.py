from typing import Dict, List

import camelot
import pandas as pd

from multi_doc_chat.logger import GLOBAL_LOGGER as log


def extract_tables_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extracts tables from a PDF file and returns them as a list of dictionaries.
    Each dictionary contains the table data in a structured format.
    """
    tables = []
    try:
        cat = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        log.info(f"PDF tables detected | file = {pdf_path} | count = {len(cat)}")

        for i, table in enumerate(cat):
            # convert the table into df:
            df: pd.DataFrame = table.df

            # xreate a csv and json based table content
            csv_text = df.to_csv(index=False)
            json_data = df.to_dict(orient="records")
 
            # append it to the list of tables
            tables.append(
                {
                    "table_index": i,
                    "page": table.page,
                    "csv": csv_text,
                    "json": json_data,
                }
            )
    except Exception as e:
        log.error(f"Error extracting tables from {pdf_path}: {e}")
    return tables


def extract_tables_from_csv(csv_path: str) -> List[Dict]:
    try:
        df = pd.read_csv(csv_path)
        return [
            {
                "csv": df.to_csv(index=False), 
                "json": df.to_dict(orient="records")
            }
        ]
    except Exception as e:
        log.warning(f"CSV read failed | file = {csv_path} |  error = {str(e)}")
        return []


def html_tables_to_json(html_path: str) -> List[Dict]:
    try:
        dfs = pd.read_html(html_path)
        tables = []
        for df in dfs:
            tables.append(
                {"csv": df.to_csv(index=False), "json": df.to_dict(orient="records")}
            )
        return tables
    except Exception as e:
        log.warning(f"HTML table extraction failed | file = {html_path} |  error = {str(e)}")
        return []
