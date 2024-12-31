import pandas as pd
import json
import os
from typing import Dict, List, Any

def flatten_parquet(input_parquet: str, parent_name: str, identifier: str) -> None:
    """
    Flattens a nested parquet file into multiple tables with relational structure.
    
    Parameters:
    - input_parquet: Path to the input parquet file.
    - parent_name: Name to use as the base for output file names (e.g., 'orders').
    - identifier: The column to use as the unique identifier for the top-level records.
    """
    # Load the original Parquet file
    try:
        df = pd.read_parquet(input_parquet)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    if identifier not in df.columns:
        print(f"Error: Identifier column '{identifier}' not found in the parquet file")
        return

    # Initialize a dictionary to store flattened data
    flattened_data: Dict[str, List[Dict[str, Any]]] = {}

    def process_nested(obj: Any, parent_keys: List[str], parent_id: Any) -> None:
        """
        Recursively processes a nested object.
        
        Parameters:
        - obj: The current object (could be a dict, list, or scalar).
        - parent_keys: A list of keys representing the current hierarchy.
        - parent_id: The identifier value from the parent object.
        """
        # Handle JSON strings
        if isinstance(obj, str):
            try:
                parsed_obj = json.loads(obj)
                if isinstance(parsed_obj, (dict, list)):
                    obj = parsed_obj
            except json.JSONDecodeError:
                pass

        if isinstance(obj, dict):
            # Create a table for this dictionary
            table_name = "_".join(parent_keys)
            if table_name not in flattened_data:
                flattened_data[table_name] = []
            
            # Add all scalar values to this table
            row_data = {f"{parent_name}_{identifier}": parent_id}
            has_scalar_values = False
            nested_items = {}
            
            # First pass: separate scalar and nested values
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    nested_items[key] = value
                else:
                    row_data[key] = value
                    has_scalar_values = True
            
            if has_scalar_values:
                # Generate a unique ID for this record if it contains an ID field
                record_id = obj.get('id') or obj.get(f'{table_name}_id') or len(flattened_data[table_name])
                row_data['id'] = record_id
                flattened_data[table_name].append(row_data)
                
                # Process nested items with the current record's ID
                for key, value in nested_items.items():
                    process_nested(value, parent_keys + [key], record_id)
            
        elif isinstance(obj, list):
            # Handle list of objects
            table_name = "_".join(parent_keys)
            if table_name not in flattened_data:
                flattened_data[table_name] = []
            
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    # Add parent reference and index
                    row_data = {
                        f"{parent_name}_{identifier}": parent_id,
                        "index": i
                    }
                    
                    # Separate scalar and nested values
                    nested_items = {}
                    for key, value in item.items():
                        if isinstance(value, (dict, list)):
                            nested_items[key] = value
                        else:
                            row_data[key] = value
                    
                    # Store the current row_data (without nested structures)
                    flattened_data[table_name].append(row_data)
                    
                    # Process nested structures with appropriate ID
                    item_id = item.get('id') or item.get(f'{table_name}_id') or f"{parent_id}_{i}"
                    for key, value in nested_items.items():
                        process_nested(value, parent_keys + [key], item_id)
                else:
                    # Handle scalar values in lists
                    flattened_data[table_name].append({
                        f"{parent_name}_{identifier}": parent_id,
                        "index": i,
                        "value": item
                    })

    # Process each row in the input dataframe
    for idx, row in df.iterrows():
        parent_id = row[identifier]
        try:
            for column_name, value in row.items():
                if pd.notna(value):  # Skip None/NaN values
                    process_nested(value, [column_name], parent_id)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Create output directory in the same location as input file
    input_dir = os.path.dirname(input_parquet)
    output_dir = os.path.join(input_dir, f"{parent_name}_flattened")
    os.makedirs(output_dir, exist_ok=True)

    # Convert flattened data to DataFrames and save as Parquet
    for table_name, rows in flattened_data.items():
        if rows:  # Skip if no data
            try:
                table_df = pd.DataFrame(rows)
                output_path = os.path.join(output_dir, f"{parent_name}_{table_name}.parquet")
                table_df.to_parquet(output_path, index=False)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error saving table {table_name}: {e}")

if __name__ == "__main__":
    # Example usage
    flatten_parquet(
        input_parquet="data/orders/orders.parquet",
        parent_name="orders",
        identifier="id"
    )
